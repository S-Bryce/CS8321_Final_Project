import spacy
from dataclasses import dataclass, field
from spacy import tokens
from warnings import warn
from fastcoref import spacy_component  # Required for the "fastcoref" component config to work
from torch import cuda
from memory_profiler import profile
import torch
import os
from openai import OpenAI
import time


# memory usage
initial_allocated = torch.cuda.memory_allocated()
initial_reserved = torch.cuda.memory_reserved()

client = OpenAI(
    api_key = "you wish"
)

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
        context_word: str = ""
        if word.pos_ == "PROPN" or word.pos_ == "NOUN":
            if word.lemma_ not in context_word:
                context_word = word.lemma_
            # Check for compound words & full names
            if word.dep_ == "compound":
                if "obj" in word.nbor(1).dep_ or word.nbor(1).dep_ == "appos":
                    context_word = word.lemma_ + " " + word.nbor(1).lemma_
                elif "obj" in word.nbor(-1).dep_ or word.nbor(-1).dep_ == "appos":
                    context_word = word.nbor(-1).lemma_ + " " + word.lemma_
            return context_word


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
        context_word: str = ""
        for word in sentence:
            if word.lemma_ not in context_word:
                context_word = word.lemma_
            if word.pos_ == "PROPN" or word.pos_ == "NOUN":
                # Check for compound words & full names
                if word.dep_ == "compound":
                    if "obj" in word.nbor(1).dep_ or word.nbor(1).dep_ == "appos":
                        sentence_context += " " + context_word
                        context_word = word.lemma_ + " " + word.nbor(1).lemma_
                        continue
                    elif "obj" in word.nbor(-1).dep_ or word.nbor(-1).dep_ == "appos":
                        context_word = word.nbor(-1).lemma_ + " " + word.lemma_
                # If the concept doesn't exist, make it
                if len(concepts) == 0 or context_word not in [concept.name for concept in concepts]:
                    concepts.append(Concept(context_word))
                # If it's the subject of the current sentence, keep track of it
                if word.dep_ == "nsubj":
                    for index, concept in enumerate(concepts):
                        if concept.name == context_word:
                            subject_index = index
                            break
                    concepts[subject_index].strength += 1
                    # Overwrite any previous context for the subject and start tracking context for the new subject
                    sentence_context = context_word
                # Otherwise, add it as a relation to the current subject
                if word.dep_ != "compound" and context_word != concepts[subject_index].name:
                    sentence_context += " " + str(word)  # Might need to use lemmatization
                    concepts[subject_index].related_concepts.append((context_word, sentence_context, global_time_index))
                    continue
            # If we have a subject, keep track of context for any connective words between it and related concepts
            if len(concepts) > 0 and subject_index != -1 and context_word != sentence_context:
                sentence_context += " " + str(word)
    return concepts


#@profile
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
            if related_concept[0] == subject or subject in related_concept[1]:
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
# example_sentence: str = ("The film follows Max Parry (Kevin Howarth), a disturbed wedding video cameraman"
#                          )
# concept_list = parse_document(example_sentence)
# print(concept_list)
#
# example_one = get_subject("Who loves coffee?")
# print(get_context(example_one, concept_list, 3, 3))
# example_two = get_subject("Where does Brandon want to travel to?")
# print(get_context(example_two, concept_list, 3, 3))
# example_three = get_subject("Who likes cats?")
# print(get_context(example_three, concept_list, 3, 3))
# example_4 = get_subject("What role does Mark Stevenson play?")
# print(get_context(example_4, concept_list, 3, 3))
# example_four = get_subject("Who portrays Max Parry?")
# print(get_context(example_four, concept_list, 3, 3))

def chatgpt_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        n=1,
        stop=["\nAnswer:"]
    )
    return response.choices[0].message["content"].strip()


def process_directory(directory_path: str):
    correct_counter = 0
    total_questions = 0

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                content = file.read()
                sections = content.split("\n\n")

                concepts = []
                for section in sections:
                    if section.startswith("PARAGRAPH"):
                        concepts = parse_document(section.split("PARAGRAPH\n")[-1])
                    else:
                        questions = [line.strip() for line in section.split("\n") if line.startswith("q:")]
                        for question in questions:
                            total_questions += 1
                            question_text = question.split("q: ")[1]
                            expected_answer = section.split(question)[1].split("a: ")[1].strip()

                            subject = get_subject(question_text)
                            context_sentences = get_context(subject, concepts, 3, 3)

                            prompt = "Using the following statements when necessary, answer the immediate question that follows. Do not include any extra information, only the answer. Each sentence in the following statements is true when read in chronological order:\n\nstatements:\n"
                            for sentence in context_sentences:
                                prompt += f". {sentence}\n"

                            prompt += f"\nquestion:\n{question_text}\n\nAnswer:"

                            answer = chatgpt_response(prompt)

                            if answer.lower() == expected_answer.lower():
                                correct_counter += 1
                            else:
                                print(f"Question: {question_text}")
                                print(f"Expected: {expected_answer}, Got: {answer}\n")
                            time.sleep(2.5)
    return correct_counter, total_questions


# Example usage:
directory_path = "C:/Users/michael.amberg/PycharmProjects/CS8321_Final_Project/temp"
correct_count, total_questions = process_directory(directory_path)
print(f"Total questions: {total_questions}")
print(f"Correct answers: {correct_count}")
print(f"Accuracy: {correct_count / total_questions * 100:.2f}%")


