import json

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
#gunhot
import redis
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import re
#gunhot

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        model_config = json.loads(args['model_config'])
        tokenizer_dir = model_config['parameters']['tokenizer_dir'][
            'string_value']
        #################### 추가 ####################
        self.client = redis.Redis(host='localhost', port=6379, db=0)
        connections.connect("default", host="localhost", port="19530")
        self.sbert_model = SentenceTransformer("/multi-e5")
        #################### 추가 ####################
        skip_special_tokens = model_config['parameters'].get(
            'skip_special_tokens')
        if skip_special_tokens is not None:
            skip_special_tokens_str = skip_special_tokens[
                'string_value'].lower()
            if skip_special_tokens_str in [
                    'true', 'false', '1', '0', 't', 'f', 'y', 'n', 'yes', 'no'
            ]:
                self.skip_special_tokens = skip_special_tokens_str in [
                    'true', '1', 't', 'y', 'yes'
                ]
            else:
                print(
                    f"[TensorRT-LLM][WARNING] Don't setup 'skip_special_tokens' correctly (set value is {skip_special_tokens['string_value']}). Set it as True by default."
                )
                self.skip_special_tokens = True
        else:
            print(
                f"[TensorRT-LLM][WARNING] Don't setup 'skip_special_tokens'. Set it as True by default."
            )
            self.skip_special_tokens = True

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                       legacy=False,
                                                       padding_side='left',
                                                       trust_remote_code=True)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Parse model output configs
        output_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT")

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get input tensors
            tokens_batch = pb_utils.get_input_tensor_by_name(
                request, 'TOKENS_BATCH').as_numpy()

            # Get sequence length
            sequence_lengths = pb_utils.get_input_tensor_by_name(
                request, 'SEQUENCE_LENGTH').as_numpy()

            # Get cum log probs
            cum_log_probs = pb_utils.get_input_tensor_by_name(
                request, 'CUM_LOG_PROBS')

            # Get sequence length
            output_log_probs = pb_utils.get_input_tensor_by_name(
                request, 'OUTPUT_LOG_PROBS')

            # Get context logits
            context_logits = pb_utils.get_input_tensor_by_name(
                request, 'CONTEXT_LOGITS')

            # Get generation logits
            generation_logits = pb_utils.get_input_tensor_by_name(
                request, 'GENERATION_LOGITS')

            # Reshape Input
            # tokens_batch = tokens_batch.reshape([-1, tokens_batch.shape[0]])
            # tokens_batch = tokens_batch.T

            # Postprocessing output data.
            outputs = self._postprocessing(tokens_batch, sequence_lengths)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            output_tensor = pb_utils.Tensor(
                'OUTPUT',
                np.array(outputs).astype(self.output_dtype))

            outputs = []
            outputs.append(output_tensor)

            if cum_log_probs:
                out_cum_log_probs = pb_utils.Tensor('OUT_CUM_LOG_PROBS',
                                                    cum_log_probs.as_numpy())
                outputs.append(out_cum_log_probs)
            else:
                out_cum_log_probs = pb_utils.Tensor(
                    'OUT_CUM_LOG_PROBS', np.array([[0.0]], dtype=np.float32))
                outputs.append(out_cum_log_probs)

            if output_log_probs:
                out_output_log_probs = pb_utils.Tensor(
                    'OUT_OUTPUT_LOG_PROBS', output_log_probs.as_numpy())
                outputs.append(out_output_log_probs)
            else:
                out_output_log_probs = pb_utils.Tensor(
                    'OUT_OUTPUT_LOG_PROBS',
                    np.array([[[0.0]]], dtype=np.float32))
                outputs.append(out_output_log_probs)

            if context_logits:
                out_context_logits = pb_utils.Tensor('OUT_CONTEXT_LOGITS',
                                                     context_logits.as_numpy())
                outputs.append(out_context_logits)
            else:
                out_context_logits = pb_utils.Tensor(
                    'OUT_CONTEXT_LOGITS', np.array([[[0.0]]],
                                                   dtype=np.float32))
                outputs.append(out_context_logits)

            if generation_logits:
                out_generation_logits = pb_utils.Tensor(
                    'OUT_GENERATION_LOGITS', generation_logits.as_numpy())
                outputs.append(out_generation_logits)
            else:
                out_generation_logits = pb_utils.Tensor(
                    'OUT_GENERATION_LOGITS',
                    np.array([[[[0.0]]]], dtype=np.float32))
                outputs.append(out_generation_logits)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=outputs)
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

    def _postprocessing(self, tokens_batch, sequence_lengths):
        outputs = []
        for batch_idx, beam_tokens in enumerate(tokens_batch):
            for beam_idx, tokens in enumerate(beam_tokens):
                # 토큰 디코딩
                seq_len = sequence_lengths[batch_idx][beam_idx]
                output = self.tokenizer.decode(tokens[:seq_len], skip_special_tokens=self.skip_special_tokens)
                #################### 추가 ####################
                print("\n" + "-" * 40 + " Llama 출력 결과 " + "-" * 40)
                print(output)
                print("-" * 90 + "\n")
                
                # 헤더 파싱 - 새로운 형식 (session_id) [action] 에 맞게 수정
                header_match = re.match(r'\(([^)]+)\)\s*\[(.*?)\]', output)

                if not header_match:
                    print("[Error] Invalid header format in output.")
                    outputs.append(b"[Error] Invalid header format.")
                    continue

                session_id, action = header_match.groups()
                action = action.strip()

                # Parsing Response
                response = output.split('assistant')[-1].strip()
                if not response:
                    print("[Error] Empty response detected.")
                    outputs.append(b"[Error] Empty response.")
                    continue

                if session_id == "-1":
                    if action == 'save':
                        response = "대화 기록이 부족하여 요약할 수 없습니다. 10개 이상의 대화가 필요합니다."
                    outputs.append(response.encode('utf8'))
                    continue

                if action == 'save':
                    print(f"[Info] Processing save operation for session {session_id}")
                    if not self._handle_save_operation(session_id, response):
                        response = "요약 저장 중 문제가 발생했습니다."
                elif action == 'clear':
                    print(f"[Info] Clearing conversation history for session {session_id}")
                    if not self.client.delete(session_id):
                        print(f"[Error] Failed to clear Redis data for session {session_id}.")
                        response = "대화 기록 정리에 실패했습니다."
                    else:
                        response = "대화 기록을 정리하였습니다."
                elif action == 'chat':
                    print(f"[Info] Storing chat response for session {session_id}")
                    if not self._store_conversation(session_id, response):
                        print(f"[Error] Failed to store chat response for session {session_id}.")
                        response = "대화 저장 중 문제가 발생했습니다."
                elif action == 'rewrite':
                    print(f"[Info] Query rewrite result for session {session_id}")
                #################### 추가 ####################
                outputs.append(response.encode('utf8'))
        
        return outputs
    
    #################### 추가 ####################
    def _handle_save_operation(self, session_id, result):
        collection_name = "collection_" + session_id
        collection = Collection(collection_name)

        summary_vector = self.sbert_model.encode([result]).tolist()[0]

        collection.load()
        search_params = {"metric_type": "IP", "params": {"nprobe": 32}}
        search_results = collection.search(
            data=[summary_vector],
            anns_field="vector",
            param=search_params,
            limit=1,
            output_fields=["text"]
        )

        similarity_threshold = 0.9
        if search_results and search_results[0][0].distance >= similarity_threshold:
            print("[Info] Similar content already exists in Milvus. Skipping save operation.")
            return False

        data = [{"vector": summary_vector, "text": result}]
        collection.insert(data)
        collection.flush()
        print(f"[Success] Saved summary to Milvus collection_{session_id}")
        return True
    #################### 추가 ####################
    #################### 추가 ####################
    def _store_conversation(self, session_id, result):
        if not self.client.ping():
            print("[Error] Redis connection is not active.")
            return False

        formatted_response = f"<|start_header_id|>assistant<|end_header_id|>{result}<|eot_id|>"
        if self.client.rpush(session_id, formatted_response.encode('utf-8')) == 0:
            print(f"[Error] Failed to push conversation to Redis for session {session_id}.")
            return False

        return True
    #################### 추가 ####################
