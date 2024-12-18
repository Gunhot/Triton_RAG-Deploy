import json
import traceback
import numpy as np
import re
import random

import triton_python_backend_utils as pb_utils
from lib.triton_decoder import TritonDecoder

class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        params = model_config['parameters']

        # 기본 설정
        self.accumulate_tokens = params.get('accumulate_tokens', {}).get('string_value', '').lower() in ['true', 'yes', '1', 't']
        self.logger = pb_utils.Logger
        
        # LLM 모델 설정
        self.llm_model_name = params.get("tensorrt_llm_model_name", {}).get("string_value", "tensorrt_llm")
        
        # 두 개의 디코더 초기화
        self.decoder = TritonDecoder(
            streaming=False,
            accumulate=self.accumulate_tokens,
            preproc_model_name="preprocessing",
            postproc_model_name="postprocessing",
            llm_model_name=self.llm_model_name)

        self.decoder2 = TritonDecoder(
            streaming=False,
            accumulate=self.accumulate_tokens,
            preproc_model_name="preprocessing2",
            postproc_model_name="postprocessing",
            llm_model_name=self.llm_model_name)

    #################### 추가 ####################
    def parse_input(self, input_text):
        match = re.match(r'\((\d+)\)\s*\[(.*?)\](.*)', input_text)
        if not match:
            raise ValueError(f"Invalid input format: {input_text}")
        return match.group(1), match.group(2).strip(), match.group(3).strip()
    #################### 추가 ####################



    #################### 추가 ####################
    def handle_chat_action(self, request, session_id, action, query):
        responses = []
        
        # talk 액션인 경우 랜덤 쿼리 선택
        if action == "talk":
            query = random.choice(["심심해", "배고파", "졸리다"])
            
        # 첫 번째 디코더에 rewrite 요청
        modified_text = f"({session_id}) [rewrite] {query}"
        req = self.decoder.convert_triton_request(request)
        req.text_input = np.array([[modified_text.encode()]], dtype=np.object_)
        
        # 첫 번째 디코더 실행
        for res in self.decoder.decode(req):
            decoded_text = res.text_output[0].decode('utf-8')
            
            # 두 번째 디코더에 chat 요청
            new_text = f"({session_id}) [{decoded_text}] {query}"

            req2 = self.decoder2.convert_triton_request(request)
            req2.text_input = np.array([[new_text.encode()]], dtype=np.object_)
            
            try:
                # 두 번째 디코더 실행
                for res2 in self.decoder2.decode(req2):
                    decoded_text2 = res2.text_output[0].decode('utf-8')
                    responses.append(self.decoder2.create_triton_response(res2))
            except Exception as e:
                print("Error in second decoder:", str(e))
                traceback.print_exc()
                
        return responses
    #################### 추가 ####################

    #################### 추가 ####################
    def handle_normal_action(self, req):
        responses = []
        for res in self.decoder.decode(req):
            responses.append(self.decoder.create_triton_response(res))
        return responses
    #################### 추가 ####################

    #################### 추가 ####################
    def process_request(self, request):
        try:
            # 입력 변환
            req = self.decoder.convert_triton_request(request)
            input_query = req.text_input[0][0].decode('utf-8')
            print(f"Input text: {input_query}")
            
            # 입력 파싱
            session_id, action, query = self.parse_input(input_query)
            
            # chat/talk 액션 처리
            if action in ["chat", "talk"]:
                return self.handle_chat_action(request, session_id, action, query)
            
            if action in ["save", "clear"]:# 일반 액션 처리
                return self.handle_normal_action(req)
            
        except Exception as e:
            self.logger.log_error(traceback.format_exc())
            return [pb_utils.InferenceResponse(
                output_tensors=[],
                error=pb_utils.TritonError(str(e)))]
        finally:
            self.decoder.reset_decoder()
            self.decoder2.reset_decoder()
    #################### 추가 ####################

    #################### 수정 ####################
    def execute(self, requests):
        responses = []
        for request in requests:
            responses.extend(self.process_request(request))
        return responses
    #################### 수정 ####################

    def finalize(self):
        print('Cleaning up...')