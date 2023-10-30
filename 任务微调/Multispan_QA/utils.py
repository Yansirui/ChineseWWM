from itertools import chain
import pickle


def is_nested_list(lst):
    if isinstance(lst, list):
        if len(lst) > 0:
            for elem in lst:
                if isinstance(elem, list):
                    return True
    return False


def subtract_offset_for_nested_list_elements(span, offset):
    if span is None:
        return [-1, -1]
    if not is_nested_list(span):
        if isinstance(span, list):
            return [elem - offset for elem in span]
        else:
            return span - offset
    else:
        return [subtract_offset_for_nested_list_elements(elem, offset) for elem in span]


def valid_nested_list_range(span, lower_bound=None, upper_bound=None):
    if not is_nested_list(span):
        if isinstance(span, list):
            for elem in span:
                if lower_bound is not None and elem < lower_bound:
                    return False
                if upper_bound is not None and elem > upper_bound:
                    return False
            return True
        else:
            if lower_bound is not None and span < lower_bound:
                return False
            if upper_bound is not None and span > upper_bound:
                return False
            return True
    else:
        for elem in span:
            if valid_nested_list_range(elem, lower_bound, upper_bound) is False:
                return False
        return True


def text_span_flatten(text_span, results):
    if text_span is None:
        return None

    if is_nested_list(text_span):
        for elem in text_span:
            text_span_flatten(elem, results)
    else:
        results.extend(text_span[:])


def leveled_text_span_flatten(text_span, level, pos, results):
    if text_span is None:
        return None

    if is_nested_list(text_span):
        for pos, elem in enumerate(text_span):
            leveled_text_span_flatten(elem, level + 1, pos, results)
    else:
        results.extend([elem + f"<level{level}><pos{pos}>" for elem in text_span])


def get_first_answer_span(answer_spans):
    if answer_spans is None:
        return [-1, -1]
    if not is_nested_list(answer_spans):
        return answer_spans
    else:
        return get_first_answer_span(answer_spans[0])


def get_flatten_answer_spans(answer_spans):
    if answer_spans is None:
        return [-1, -1]
    if not is_nested_list(answer_spans):
        return answer_spans
    else:
        res = []
        for elem in answer_spans:
            if is_nested_list(elem):
                res += get_flatten_answer_spans(elem)
            elif isinstance(elem, list):
                res += elem
            else:
                res += [elem]
        return res


def get_first_answer_span_text(answer_span_texts):
    if answer_span_texts is None:
        return None
    if isinstance(answer_span_texts, str):
        return answer_span_texts
    else:  # non-empty list
        return get_first_answer_span_text(answer_span_texts[0])


def get_flatten_answer_span_texts(answer_span_texts):
    if answer_span_texts is None:
        return None
    if not is_nested_list(answer_span_texts):
        return answer_span_texts
    else:
        res = []
        for elem in answer_span_texts:
            if is_nested_list(elem):
                res += get_flatten_answer_spans(elem)
            elif isinstance(elem, list):
                res += elem
            else:
                res += [elem]
        return res


def nested_list_shape_match_or_not(a, b):
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        elif len(a) != 0:
            return nested_list_shape_match_or_not(a[0], b[0]) and nested_list_shape_match_or_not(a[1:], b[1:])
    else:
        return True


def match_answer_span_and_text(answer_span, answer_text):
    if answer_text is None and (answer_span is None or answer_span == [-1, -1]):
        return True
    l1 = len(get_flatten_answer_spans(answer_span)) // 2
    l2 = len(get_flatten_answer_span_texts(answer_text))
    return l1 == l2


def built_filtered_data(input_file, output_file):
    orig_sample = pickle.load(open(input_file, "rb"))
    invalids = [i for i, example in enumerate(orig_sample) if
                not match_answer_span_and_text(example['answer_span'], example['answer_text'])]
    filtered_examples = [example for i, example in enumerate(orig_sample) if i not in invalids]
    pickle.dump(filtered_examples, open(output_file, 'wb'))
