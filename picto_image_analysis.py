import json
import os
import requests
import base64
import tempfile

# LangchainRecommender 클래스 (코드 내 analyze_image 메소드 사용)
from langchain_recommender import LangchainRecommender  # 미리 해당 파일에 클래스가 정의되어 있다고 가정

# JSON 파일 경로와 업데이트된 JSON 저장 경로
JSON_FILE = './json.txt'
UPDATED_JSON_FILE = './json_updated.txt'

# 사진 다운로드 URL 템플릿
URL_TEMPLATE = "http://bogota.iptime.org:8086/photo-store/photos/download/{}"

def download_image(photo_id):
    """photo_id를 사용해 이미지를 다운로드하고 임시 파일 경로를 반환합니다."""
    url = URL_TEMPLATE.format(photo_id)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # 임시 파일 생성 (JPEG 파일)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        with open(temp_file.name, 'wb') as f:
            f.write(response.content)
        return temp_file.name
    else:
        print(f"사진 {photo_id} 다운로드 실패, status_code: {response.status_code}")
        return None

def main():
    # JSON 파일 로드
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # LangchainRecommender 인스턴스 생성
    recommender = LangchainRecommender()
    
    # 각 레코드 처리
    for record in data:
        photo_id = record.get('photo_id')
        print(f"photo_id {photo_id} 처리 시작")
        image_path = download_image(photo_id)
        if not image_path:
            record['content'] = "이미지 다운로드 실패"
            continue
        
        try:
            # analyze_image 메소드 사용해 이미지 분석 결과 얻기
            analysis_text = recommender.analyze_image(image_path)
            record['content'] = analysis_text
            print(f"content : {analysis_text}")
            print(f"photo_id {photo_id} 분석 완료")
        except Exception as e:
            print(f"photo_id {photo_id} 분석 중 오류 발생: {str(e)}")
            record['content'] = f"분석 오류: {str(e)}"
        finally:
            # 임시 이미지 파일 삭제
            if os.path.exists(image_path):
                os.remove(image_path)
    
    # 업데이트된 결과를 JSON 파일로 저장
    with open(UPDATED_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"업데이트된 JSON 파일이 {UPDATED_JSON_FILE}에 저장되었습니다.")

if __name__ == "__main__":
    main()
