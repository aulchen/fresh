import torch
from heapq import *

class BertExtractor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    '''
    Given a 1d tensor representing either the input_ids or attention, remove the [CLS] and [SEP] tokens.
    Assumed that [CLS] and [SEP] are at the start and end of the query
    '''

    def trim_1d_tensor(tensor) -> object:
        length = len(tensor)
        return tensor[1:length - 1]

    '''
    Given truncated input_ids stripped of attention masks, return the text that represents them.
    '''
    def input_ids_to_text(self, input_ids):
        input_ids = torch.squeeze(input_ids)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        return " ".join(tokens)

    '''
    Given a text query, get trimmed input_ids and trimmed average attentions
    '''
    def get_input_ids_attention(self, query):
        encoding = self.tokenizer(query,
                             return_tensors='pt',
                             max_length=512,
                             truncation='longest_first',
                             )
        input_ids_trim = BertExtractor.trim_1d_tensor(encoding['input_ids'].flatten())
        # Get feature attention scores
        with torch.no_grad():
            model_output = self.model(**encoding,
                                      output_attentions=True)
        # Last layer (-1), only element in the batch (0), all heads (:), from CLS token (0) to all others (:)
        avg_attention = torch.mean(model_output.attentions[-1][0, :, 0, :], dim=0)
        # Remove attentions to/from CLS and SEP tokens
        avg_attention_trim = BertExtractor.trim_1d_tensor(avg_attention)
        return (input_ids_trim, avg_attention_trim)


class BertTopKExtractor(BertExtractor):
    def __init__(self, model, tokenizer, k):
        super().__init__(model, tokenizer)
        self.k = k

    # Given single query, return the tokens needed for Bert
    def extract(self, query):
        input_ids, avg_attention = self.get_input_ids_attention(query)
        # If the length of the input ids<= k, return input_ids converted back to text
        if len(input_ids) <= self.k:
            input_ids = torch.cat([torch.tensor([101]), input_ids, torch.tensor([102])])
            input_ids = torch.reshape(input_ids, (1, -1))
            return input_ids

        # Use a heap to build a priority queue
        h = []
        assert len(input_ids) == len(avg_attention)
        for index in range(0, len(input_ids)):
            # Priority, input_id, index
            heappush(h, (-avg_attention[index].item(), input_ids[index].item(), index))
        # Pop the top k elements off of the queue
        output = []
        for _ in range(0, self.k):
            output.append(heappop(h))

        # Return sorted based on index 2, the original ordering
        output = sorted(output, key=lambda x: x[2])
        # Extract only the input_ids from the output
        input_ids = list(zip(*output))[1]
        input_ids = list(input_ids)
        # Add the CLS and SEP tokens
        input_ids = [101] + input_ids + [102]
        # Convert into 2D tensor for compatibility with BERT
        input_ids = torch.tensor([input_ids])
        return input_ids


class BertContiguousKExtractor(BertExtractor):
    def __init__(self, model, tokenizer, k):
        super().__init__(model, tokenizer)
        self.k = k

    def extract(self, query):
        input_ids, avg_attention = self.get_input_ids_attention(query)
        # If the length of the input ids<= k, return input_ids padded again
        if len(input_ids) <= self.k:
            input_ids = torch.cat([torch.tensor([101]), input_ids, torch.tensor([102])])
            input_ids = torch.reshape(input_ids, (1, -1))
            return input_ids
        # 1D convolution of length k and weight 1 with the avg_attentions
        # The argmax of the result is the start index of the rationale

        # Reshape avg_attentions to a 3D shape
        avg_attention_3d = torch.reshape(avg_attention, (1, 1, -1))
        weight = torch.ones((1, 1, self.k))
        # Do the convolution
        attention_conv = torch.nn.functional.conv1d(avg_attention_3d, weight).squeeze()
        # Find the argmax
        start_index = torch.argmax(attention_conv).item()
        # Extract the input_ids of the rationale
        input_ids = input_ids[start_index: start_index + self.k]
        # pad the result
        input_ids = torch.cat([torch.tensor([101]), input_ids, torch.tensor([102])])
        input_ids = torch.reshape(input_ids, (1, -1))
        return input_ids