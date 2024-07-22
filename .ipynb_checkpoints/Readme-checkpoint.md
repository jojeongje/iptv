## ixi-gen 후처리 모듈

ixi_gen_postprocess_executor 함수를 호출해주시면 됩니다.
ixi_gen_postprocess_executor 함수에서는 request_body와 response_body 전문을 입력 으로 받아서
response_body를 후처리하여 리턴하게 됩니다.

ixi_gen_postprocess_executor 내부에서는 "reduce_summary_keywords", "perform_hallucination_check"
내부 함수를 호출하게 되는데 현재는 임의로 작성해두었습니다.

reduce_summary_keywords는 상세 요약문 및 키워드 개수를 줄이는 기능을 하는 함수이고
perform_hallucination_check는 AI Task 제안에서의 Hallucination을 확인하여 유지/제거하는 기능을 하는 함수입니다.

기타 다른 함수가 추가될 수 있으나, 다른 함수들도 ixi_gen_postprocess_executor 함수 내부에서 동작할 것임으로 ixi_gen_postprocess_executor를 호출하여 활용해주시면 됩니다.
