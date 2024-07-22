import json
from collections import defaultdict
import numpy as np
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer


def cal_keyword_score(tfidf_matrix, feature_names, input_segment, keyword):
    total_score = []
    full_stt = "".join((" ".join(input_segment)).split())
    back_n_front_idx = [i for i in range(round(len(keyword)*0.2))] + [i for i in range(round(len(keyword)*0.8), len(keyword))]
    for i, segment in enumerate(keyword):
        if segment != []:
            segment_score_dict = defaultdict(float)
            for k in segment:
                k = k.replace("#","")
                # tf-idf 스코어 합산
                if k in feature_names:
                    keyword_index = feature_names.tolist().index(k)
                    tfidf_score = tfidf_matrix[i, keyword_index]
                    segment_score_dict[k] = tfidf_score
                else:
                    segment_score_dict[k] = 0
                # 전체 콜 음절 단위 등장 빈도 계산
                segment_score_dict[k] += (full_stt.count(k) / 100)
                # 세그먼트 길이 점수 합산
                segment_score_dict[k] += (len(input_segment[i]) / 1000)
                # 처음 문맥,끝 문맥 추가 점수 합산
                if i in back_n_front_idx:
                    segment_score_dict[k] = segment_score_dict[k] * 1.5
            segment_score_dict = dict(sorted(segment_score_dict.items(), key=lambda x:x[1], reverse=True))
            total_score.append(segment_score_dict)
    return total_score

def do_extract_keyword(result, main_keyword, total_keyword_num=8):
    final_result = []
    new_result = []
    for k in result:
        new_result.append([(a,b) for a,b in k[0].items()])
    while total_keyword_num != 0:
        for k_list in new_result:
            if k_list == []:
                continue
            if total_keyword_num == 0:
                break
            score_list = [v for _,v in k_list]
            std_dev = np.std(score_list)
            # 점수 분산이 크면 top ranking 추출
            if std_dev >= 0.02:
                for keyword, _ in k_list:
                    if not keyword in final_result:
                        final_result.append(keyword)
                        break
            # 점수 분산이 작으면 랜덤 추출
            else:
                keyword_list = [k for k,_ in k_list]
                temp = list(set(keyword_list)-set(final_result))
                sample = random.choice(temp)
                final_result.append(sample)
            total_keyword_num -= 1
    final_result = main_keyword + ["#"+result for result in final_result]
    return final_result

def cal_summary_score(tfidf_matrix, feature_names, input_segment, summary):
    total_score = []
    full_stt = "".join((" ".join(input_segment)).split())
    back_n_front_idx = [i for i in range(round(len(summary)*0.2))] + [i for i in range(round(len(summary)*0.8), len(summary))]
    for i, segment in enumerate(summary):
        if segment != []:
            segment_score_dict = defaultdict(float)
            for k in segment:
                k = k.strip()
                for s in k.split():
                    # tf-idf 스코어 합산
                    if s in feature_names:
                        summary_index = feature_names.tolist().index(s)
                        tfidf_score = tfidf_matrix[i, summary_index]
                        segment_score_dict[k] += tfidf_score
                    # 전체 콜 음절 단위 등장 빈도 계산
                    segment_score_dict[k] += (full_stt.count(s) / 100)
                    # 세그먼트 길이 점수 합산
                    segment_score_dict[k] += (len(input_segment[i]) / 1000)
                    # 처음 문맥,끝 문맥 추가 점수 합산
                    if i in back_n_front_idx:
                        segment_score_dict[k] = segment_score_dict[k] * 1.5
            segment_score_dict = dict(sorted(segment_score_dict.items(), key=lambda x:x[1], reverse=True))
            total_score.append(segment_score_dict)
    return total_score

def do_extract_summary(result, total_summary_num=20):
    final_result = []
    new_result = []
    for k in result:
        new_result.append([(a,b) for a,b in k[0].items()])
    while total_summary_num != 0:
        for k_list in new_result:
            if k_list == []:
                continue
            if total_summary_num == 0:
                break
            score_list = [v for _,v in k_list]
            std_dev = np.std(score_list)
            # 점수 분산이 크면 top ranking 추출
            if std_dev >= 1.5:
                for summary, _ in k_list:
                    if not summary in final_result:
                        final_result.append(summary)
                        break
            # 점수 분산이 작으면 랜덤 추출
            else:
                summary_list = [k for k,_ in k_list]
                temp = list(set(summary_list)-set(final_result))
                sample = random.choice(temp)
                final_result.append(sample)
            total_summary_num -= 1
    final_result =  [result for result in final_result]
    return final_result

# 키워드나 요약 중 비어있는 세그먼트 처리
# 요약쪽 함수 생성 및 테스트

def process(paragraph_info):
    input_segment = [stt for stt, s, k in paragraph_info] # STT문장 내 개행문자 제거 및 띄어쓰기 중복 제거
    input_segment_extract = []
    for stt_list in input_segment:
        temp = []
        for stt in stt_list:
            temp.append(stt[stt.index(':')+1:].strip())
        input_segment_extract.append(" ".join(temp))
    input_segment = input_segment_extract
    keyword = [k for stt, s, k in paragraph_info]
    summary_detail = [s for stt, s, k in paragraph_info]

    # 세그먼트별 TF-IDF 계산
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(input_segment)
    feature_names = vectorizer.get_feature_names_out()

    #핵심 키워드 추출 및 중복 키워드 제거
    main_keyword_type = ['#일상', '#업무', '#예약', '#약속', '#문의']
    main_keyword = []
    for idx,key in enumerate(keyword):
        for k in key:
            if k in main_keyword_type:
                main_keyword.append(key.pop(key.index(k)))
        keyword[idx] = list(set(key))

    main_keyword_list = list(set(main_keyword))

    main_keyword_count = [(k, main_keyword.count(k)) for k in main_keyword_list]
    main_keyword = sorted(main_keyword_count, key=lambda x:x[1], reverse=True)
    main_keyword = [k for k,v in main_keyword[:2]] # 빈도많은 2개의 핵심 키워드 추출

    # 전체 키워드 개수 파악
    keyword_temp = []
    for k in keyword:
        keyword_temp.extend(k)
    keyword_temp = list(set(keyword_temp))
    if len(keyword_temp) + len(main_keyword) <= 10:
        keyword = main_keyword + keyword_temp
    else:
        keyword_score = cal_keyword_score(tfidf_matrix, feature_names, input_segment, keyword)
        # 합산 점수 계산
        result = []
        for k in keyword_score:
            score = 0
            for v in k.values():
                score += v
            result.append([k,score])
        result = sorted(result, key=lambda x:x[1], reverse=True)
        keyword = do_extract_keyword(result, main_keyword, total_keyword_num=10-len(main_keyword))
    
    # 전체 요약 개수 파악
    summary_detail_temp = []
    for i,s in enumerate(summary_detail):
        summary_detail[i] = list(set(s))
        summary_detail_temp.extend(summary_detail[i])
    summary_detail_temp = list(set(summary_detail_temp))
    if len(summary_detail_temp) <= 20:
        summary_detail = summary_detail_temp
    else:
        summary_score = cal_summary_score(tfidf_matrix, feature_names, input_segment, summary_detail)
        # 합산 점수 계산
        result = []
        for k in summary_score:
            score = 0
            for v in k.values():
                score += v
            result.append([k,score])
        result = sorted(result, key=lambda x:x[1], reverse=True)
        summary_detail = do_extract_summary(result)

    return (keyword, summary_detail)


def main():
    with open('./long_stt_infer_result.json') as f:
        input_file = json.load(f)

    with open('./long_stt_sample.txt') as f:
        input_stt = f.readlines()
    input_stt = [s.strip() for s in input_stt]


    # 세그먼트 나누기(입력으로 필요)
    input_segment = []
    temp = []
    for idx,stt in enumerate(input_stt):
        if len(" ".join(temp) + stt + " ") < 1300:
            temp.append("${}".format(idx+1)+" "+stt)
        else:
            input_segment.append(temp)
            temp = []
            temp.append("${}".format(idx+1)+" "+stt)
    if temp != []:
        input_segment.append(temp)

    print(input_segment)
    keyword = input_file["keyword"]
    summary = input_file["summary_detail"]

    # 결과가 비어있는 경우 테스트
    # for i,k in enumerate(keyword):
    #     if i == 0 or i == 3 or i == 10:
    #         keyword[i] = []
    # for i,s in enumerate(summary):
    #     if i == 0 or i == 3 or i == 10:
    #         summary[i] = []

    paragraph_info = [(stt,s,k) for stt, k, s in zip(input_segment, keyword, summary)]

    # 입력
    # paragraph_info = [(세그먼트STT리스트1, 요약결과리스트1, 키워드결과리스트1,), (세그먼트STT리스트2, 요약결과리스트2, 키워드결과리스트2), ...]

    # 출력
    # keyword = [keyword1, keyword2, ... , keyword10] # 10개 미만 키워드(+맨 앞 2개 핵심 키워드)
    # summary_detail = [summary1, summary2, ... , summary20] # 20개 미만 요약
    # 단락 등장 순서 고려하지 않고 추출
        # 예시 : 입력 (단락1, 단락2, 단락3 순서) --> 출력 (단락2, 단락1, 단락3 순서)

    start = time.time()
    keyword, summary_detail = process(paragraph_info)
    print("소요 시간 :",time.time()-start)
    print("키워드 결과 : ",keyword)
    print("요약 결과 : ", summary_detail)

if __name__ == "__main__":
    main()


