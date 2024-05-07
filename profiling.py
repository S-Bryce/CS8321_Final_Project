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

import psutil
import time
import tracemalloc
from memory_profiler import memory_usage

@dataclass
class Concept:
    name: str
    strength: int = 0
    related_concepts: list[tuple[str, str, int]] = field(default_factory=lambda: [])

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

class MetricsTracker:
    def __init__(self):
        self.metrics = {}
        self.total_start_time = None
        self.total_memory_start = None
        self.process = psutil.Process()

    def start(self, name):
        self.metrics[name] = {
            'start_time': time.time(),
            'start_memory': self.process.memory_info().rss / (1024 ** 2)
        }

    def stop(self, name):
        end_time = time.time()
        end_memory = self.process.memory_info().rss / (1024 ** 2)
        start_time = self.metrics[name]['start_time']
        start_memory = self.metrics[name]['start_memory']
        self.metrics[name]['duration'] = end_time - start_time
        self.metrics[name]['memory_usage'] = end_memory - start_memory

    def start_total(self):
        self.total_start_time = time.time()
        self.total_memory_start = self.process.memory_info().rss / (1024 ** 2)

    def stop_total(self):
        total_end_time = time.time()
        total_end_memory = self.process.memory_info().rss / (1024 ** 2)
        return {
            'total_duration': total_end_time - self.total_start_time,
            'total_memory_usage': total_end_memory - self.total_memory_start
        }

    def print_metrics(self):
        for name, metrics in self.metrics.items():
            print(f'{name} - Duration: {metrics["duration"]:.4f}s, Memory Usage: {metrics["memory_usage"]:.4f}MB')

tracker = MetricsTracker()

def get_subject(sentence: str) -> str:
    tracker.start('get_subject')
    resolved_sentence = coref_resolver(sentence.lower(), component_cfg={"fastcoref": {"resolve_text": True}})
    processed_sentence = language_processor(resolved_sentence._.resolved_text)

    for word in processed_sentence:
        context_word = ""
        if word.pos_ == "PROPN" or word.pos_ == "NOUN":
            if word.lemma_ not in context_word:
                context_word = word.lemma_
            if word.dep_ == "compound":
                if "obj" in word.nbor(1).dep_ or word.nbor(1).dep_ == "appos":
                    context_word = word.lemma_ + " " + word.nbor(1).lemma_
                elif "obj" in word.nbor(-1).dep_ or word.nbor(-1).dep_ == "appos":
                    context_word = word.nbor(-1).lemma_ + " " + word.lemma_
            tracker.stop('get_subject')
            return context_word

    tracker.stop('get_subject')
    return ""

def parse_document(document: str) -> list[Concept]:
    tracker.start('parse_document')
    resolved_document = coref_resolver(document.lower(), component_cfg={"fastcoref": {"resolve_text": True}})
    processed_document = language_processor(resolved_document._.resolved_text)

    global_time_index = -1
    concepts = []
    for sentence in processed_document.sents:
        global_time_index += 1
        subject_index = -1
        sentence_context = ""

        context_word = ""
        for word in sentence:
            if word.lemma_ not in context_word:
                context_word = word.lemma_
            if word.pos_ == "PROPN" or word.pos_ == "NOUN":
                if word.dep_ == "compound":
                    if "obj" in word.nbor(1).dep_ or word.nbor(1).dep_ == "appos":
                        sentence_context += " " + context_word
                        context_word = word.lemma_ + " " + word.nbor(1).lemma_
                        continue
                    elif "obj" in word.nbor(-1).dep_ or word.nbor(-1).dep_ == "appos":
                        context_word = word.nbor(-1).lemma_ + " " + word.lemma_
                if len(concepts) == 0 or context_word not in [concept.name for concept in concepts]:
                    concepts.append(Concept(context_word))
                if word.dep_ == "nsubj":
                    for index, concept in enumerate(concepts):
                        if concept.name == context_word:
                            subject_index = index
                            break
                    concepts[subject_index].strength += 1
                    sentence_context = context_word
                if word.dep_ != "compound" and context_word != concepts[subject_index].name:
                    sentence_context += " " + str(word)
                    concepts[subject_index].related_concepts.append((context_word, sentence_context, global_time_index))
                    continue
            if len(concepts) > 0 and subject_index != -1 and context_word != sentence_context:
                sentence_context += " " + str(word)
    tracker.stop('parse_document')
    return concepts

def get_context(subject: str, concepts: list[Concept], num_context: int = 3, recency_multiplier: int = 3) -> list[str]:
    tracker.start('get_context')

    def extract_score(sample: tuple[str, int]) -> int:
        return sample[1]

    context_samples = []

    for concept in concepts:
        for related_concept in concept.related_concepts:
            if related_concept[0] == subject or subject in related_concept[1]:
                context_samples.append((related_concept[1], related_concept[2] * recency_multiplier + concept.strength))

    for concept in concepts:
        if concept.name == subject:
            for related_concept in concept.related_concepts:
                context_samples.append((related_concept[1], related_concept[2] * recency_multiplier + concept.strength))

    if num_context > len(context_samples):
        num_context = len(context_samples)

    context = [context[0] for context in sorted(context_samples, key=extract_score)][:num_context]
    tracker.stop('get_context')
    return context


def process_directory(directory_path):
    tracker.start_total()
    correct_counter = 0
    total_questions = 0

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                content = file.read()
                sections = content.split("\n\n")

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
                            ans = get_context(subject, concepts, 1, 1)
                            print(ans)
                            if ans == expected_answer:
                                correct_counter += 1

    total_metrics = tracker.stop_total()
    print(f"Overall Program - Duration: {total_metrics['total_duration']:.4f}s, Memory Usage: {total_metrics['total_memory_usage']:.4f}MB")
    tracker.print_metrics()

    return correct_counter, total_questions

# Example usage
directory_path = "C:/Users/michael.amberg/PycharmProjects/CS8321_Final_Project/temp"
correct, total = process_directory(directory_path)
print(f"Correct: {correct}/{total}")



