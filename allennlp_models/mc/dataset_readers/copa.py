
from allennlp.common.file_utils import cached_path, logger, json_lines_from_file
from allennlp.data import DatasetReader
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from overrides import overrides

from allennlp_models.mc.dataset_readers.transformer_mc import TransformerMCReader

@DatasetReader.register("copa")
class CopaReader(TransformerMCReader):
    def __init__(self, transformer_model_name: str = "roberta-large", length_limit: int = 512, **kwargs
                 ) -> None:
        super().__init__(**kwargs)

        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False
        )
        self._token_indexers = {"tokens": PretrainedTransformerIndexer(transformer_model_name)}
        self.length_limit = length_limit

    @overrides
    def _read(self, file_path):
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)

        for json in json_lines_from_file(file_path):
            choices = [json["choice1"], json["choice2"]]

            if json["question"].upper() == "CAUSE":
                sub_sent = "What was the CAUSE of this?"

            elif json["question"].upper() == "EFFECT":
                sub_sent = "What happened as a RESULT?"
            else:
                pass

            premise = json["premise"] + sub_sent

            #choices = [(choice["label"], choice["text"]) for choice in json["question"]["choices"]]
            # correct_choice = [
            #     i for i, (label, _) in enumerate(choices) if label == json["answerKey"]
            # ][0]
            yield self.text_to_instance(
                json["idx"], premise, [c for c in choices], json["label"]
            )
