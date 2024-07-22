import json
with open('request_body.json', 'r') as f:
    request_body = json.load(f)

import json
with open('response_body.json', 'r') as f:
    response_body = json.load(f)

from typing import Dict, Any
def ixi_gen_postprocess_executor(request_body: Dict[str, Any], response_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main executor function for postprocessing ixi-gen output.
    This function performs two main tasks
    1. Reduces the number of summary/keywords
    2. Performs a hallucination check

    Args:
        request_body (Dict[str, Any]): JSON object containing the request body.
        response_body (Dict[str, Any]): JSON object containing the response body.

    Returns:
        Dict[str, Any]: Postprocessed data.
    """
    stt_dialogue = request_body['modelParams']['text']["input"]
    response_body = _reduce_summary_keywords(stt_dialogue, response_body)
    response_body = _perform_hallucination_check(stt_dialogue, response_body)
    return response_body

def _reduce_summary_keywords(stt_dialogue: Dict[str, Any], response_body: Dict[str, Any]):
    """
    Reduces the number of detail summaries/keywords
    
    Args:
        request_body (Dict[str, Any]): JSON object containing the request body.
        response_body (Dict[str, Any]): JSON object containing the response body.

    Returns:
        Dict[str, Any]: Data with reduced summary/keywords.
    """
    import random
    num_select = 3 
    selected_summary = random.sample(response_body["result"]["summary"]["summaryDetail"], num_select)
    response_body["result"]["summary"]["summaryDetail"] = selected_summary

    num_select = 3 
    selected_keyword = random.sample(response_body["result"]["keyword"], num_select)
    response_body["result"]["keyword"] = selected_keyword
    return response_body

def _perform_hallucination_check(stt_dialogue: Dict[str, Any], response_body: Dict[str, Any]):
    """
    Performs a hallucination check
    
    Args:
        request_body (Dict[str, Any]): JSON object containing the request body.
        response_body (Dict[str, Any]): JSON object containing the response body.

    Returns:
        Dict[str, Any]: Data with hallucinations checked
    """
    checked_entities = []
    for entities in response_body["result"]["taskRecommend"]:
        checked_entity = {}
        if entities == {}:
            checked_entities.append(checked_entity)
            continue
            
        else:
            if "호수" not in list(entities.values()):
                checked_entity = entities
                checked_entities.append(checked_entity)
    response_body["result"]["taskRecommend"] = checked_entities
    return response_body


if __name__=="__main__":
    response_body = ixi_gen_postprocess_executor(request_body, response_body)
    pprint(response_body,sort_dicts=False)
