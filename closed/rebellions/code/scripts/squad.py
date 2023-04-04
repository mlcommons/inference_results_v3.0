from json import load
from transformers import BertTokenizer
import collections
import numpy as np
import os


class SquadExample(object):
    def __init__(self, qas_id, question_text, doc_tokens):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        return s


def _check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def tokenize(data_file: str, vocab_file: str, cache_file: str) -> None:
    if os.path.exists(cache_file):
        return
    tokenizer = BertTokenizer(vocab_file)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    with open(data_file) as f:
        data = load(f)
    features = []
    for topic in data["data"]:
        for paragraph in topic["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                example = SquadExample(
                    qas_id=qas_id, question_text=question_text, doc_tokens=doc_tokens
                )
                features.append(example)

    features = features[0:]
    final_samples = []
    max_query_length = 64
    doc_stride = 128
    max_seq_length = 384

    for example_index, example in enumerate(features):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        all_doc_tokens = []
        for i, token in enumerate(example.doc_tokens):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for doc_span_index, doc_span in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)

            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)

            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            final_samples.append(list((input_ids, segment_ids, input_mask)))

    np_samples = np.array(final_samples, dtype=np.int32)
    np.save(cache_file, np_samples)
    return final_samples
