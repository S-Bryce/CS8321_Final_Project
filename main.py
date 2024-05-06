import spacy
from dataclasses import dataclass, field
from spacy import tokens
from warnings import warn
from fastcoref import spacy_component  # Required for the "fastcoref" component config to work
from torch import cuda

if not cuda.is_available():
    warn("Using CPU for inference", RuntimeWarning)

# Alternatives: en_core_web_md, en_core_web_lg, en_core_web_trf
# Don't forget to download the models! "python -m spacy download <model>"
coref_resolver: spacy.Language = spacy.load("en_core_web_sm", exclude=["ner", "textcat", "paser", "lemmatizer"])
# You can delete the config= stuff if you want to use a faster (but less accurate) system.
# NOTE: You will need to download the model.bin file from
# https://huggingface.co/biu-nlp/lingmess-coref/resolve/main/pytorch_model.bin?download=true and place it into the
# lingmess-coref folder.
coref_resolver.add_pipe("fastcoref", config={"model_architecture": "LingMessCoref",
                                             "model_path": "lingmess-coref",
                                             "device": "cuda" if cuda.is_available() else "cpu"})
language_processor: spacy.Language = spacy.load("en_core_web_sm")


@dataclass
class Concept:
    name: str
    strength: int = 0
    related_concepts: list[tuple[str, str, int]] = field(default_factory=lambda: [])


def get_subject(sentence: str) -> str:
    resolved_sentence: tokens.Doc = coref_resolver(sentence.lower(),
                                                   component_cfg={"fastcoref": {"resolve_text": True}})
    processed_sentence: tokens.Doc = language_processor(resolved_sentence._.resolved_text)

    for word in processed_sentence:
        # Ignore pronouns to hopefully answer questions such as "Who likes cats?" correctly.
        if word.pos_ == "PROPN" or word.pos_ == "NOUN":
            return word.lemma_


def parse_document(document: str) -> list[Concept]:
    resolved_document: tokens.Doc = coref_resolver(document.lower(),
                                                   component_cfg={"fastcoref": {"resolve_text": True}})
    processed_document: tokens.Doc = language_processor(resolved_document._.resolved_text)

    sentence: tokens.Span
    global_time_index: int = -1
    concepts: list[Concept] = []
    for sentence in processed_document.sents:
        global_time_index += 1
        subject_index: int = -1
        sentence_context: str = ""

        word: tokens.Token
        for word in sentence:
            if word.pos_ == "PROPN" or word.pos_ == "NOUN":
                # If the concept doesn't exist, make it
                if len(concepts) == 0 or word.lemma_ not in [concept.name for concept in concepts]:
                    concepts.append(Concept(word.lemma_))
                # If it's the subject of the current sentence, keep track of it
                if word.dep_ == "nsubj":
                    for index, concept in enumerate(concepts):
                        if concept.name == word.lemma_:
                            subject_index = index
                            break
                    concepts[subject_index].strength += 1
                    # Overwrite any previous context for the subject and start tracking context for the new subject
                    sentence_context = word.lemma_
                # Otherwise, add it as a relation to the current subject
                if word.dep_ != "compound" and word.lemma_ != concepts[subject_index].name:
                    # TODO: Combine pobj with compound word.nbor(-1)
                    sentence_context += " " + str(word)  # Might need to use lemmatization
                    concepts[subject_index].related_concepts.append((word.lemma_, sentence_context, global_time_index))
            # If we have a subject, keep track of context for any connective words between it and related concepts
            if len(concepts) > 0 and subject_index != -1 and word.lemma_ != sentence_context:
                sentence_context += " " + str(word)
    return concepts


def get_context(subject: str, concepts: list[Concept], num_context: int = 3, recency_multiplier: int = 3) -> list[str]:
    """

    :param subject: Lemmatized form of the subject from the original query sentence.
    :param concepts: list of Concept objects related to the corpus.
    :param num_context: The number of context items to be returned. Default of 3.
    :param recency_multiplier: Multiplicative weight for overall context score calculation. Default of 3.
    :return: List of contextually relevant strings, ordered by score.
    """

    # Could be a lambda function, but then type-hinting would trigger variable shadowing false positives
    def extract_score(sample: tuple[str, int]) -> int:
        return sample[1]

    context_samples: list[tuple[str, int]] = []

    # In the general case, we can expect a question in the form of "Who wants to travel to Paris?", in which case we can
    # find related concepts and then score them by the strength of the original concept (how frequently that concept was
    # mentioned) combined with the recentness of the relation.
    concept: Concept
    for concept in concepts:

        related_concept: tuple[str, str, int]
        for related_concept in concept.related_concepts:
            if related_concept[0] == subject:
                context_samples.append((related_concept[1], related_concept[2] * recency_multiplier + concept.strength))

    # If we're dealing with a sentence where the subject is not the concept in question, e.g., "Where does Brandon want
    # to travel?", then we search matching top-level concepts and give a best-effort list of relations.
    for concept in concepts:
        if concept.name == subject:
            for related_concept in concept.related_concepts:
                context_samples.append((related_concept[1], related_concept[2] * recency_multiplier + concept.strength))

    if num_context > len(context_samples):
        num_context = len(context_samples)

    # Returns list by ascending relevancy score, since inputs lower in an LLM's prompt are more significant to it.
    context: tuple[str, int]
    return [context[0] for context in sorted(context_samples, key=extract_score)][:num_context]


# This would be where your data gets loaded!
example_sentence: str = ("Brandon loves coffee. He wants to travel to Paris. He likes cats, and cats love him. Now, "
                         "Brandon no longer likes them.")
concept_list = parse_document(example_sentence)
print(concept_list)

example_one = get_subject("Who loves coffee?")
print(get_context(example_one, concept_list, 3, 3))
example_two = get_subject("Where does Brandon want to travel to?")
print(get_context(example_two, concept_list, 3, 3))
example_three = get_subject("Who likes cats?")
print(get_context(example_three, concept_list, 3, 3))
