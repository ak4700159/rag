from langchain_recommender import LangchainRecommender
from config import DEFAULT_IMAGE_PATH, DEFAULT_PDF_PATH
import os

# 경로 설정
IMAGES_PATH = os.path.join(os.path.dirname(__file__), "ImagesPath")
DEFAULT_IMAGE_PATH = os.path.join(IMAGES_PATH, "land.jpg")

def chat_with_gpt(recommender, user_input):
    """GPT와 대화하는 함수"""
    try:
        response = recommender.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 사진 촬영과 관련된 질문에 대해 전문적으로 답변해주세요."
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT 응답 중 오류가 발생했습니다: {str(e)}"

def main():
    # 추천 시스템 초기화
    recommender = LangchainRecommender()

    print("\n=== 이미지 사진 분석 시스템 ===")
    print("시스템을 초기화합니다...")

    # PDF 처리 확인
    if not os.path.exists(DEFAULT_PDF_PATH):
        print(f"기본 PDF 파일을 찾을 수 없습니다: {DEFAULT_PDF_PATH}")
        print("PDF 파일을 먼저 처리해주세요.")
        return

    print("\nPDF 처리 중...")
    if recommender.process_pdf(DEFAULT_PDF_PATH):
        print("PDF 처리가 완료되었습니다.")
    else:
        print("PDF 처리에 실패했습니다.")
        return
    """픽토리 (Pictory) → Picture + History / Story"""
    print("\n안녕하세요🤚 저는 AI 픽토리 입니다 무엇을 도와드릴까요?.😊")

    print("\n")
    print("픽토리는 다음과 같은 기능을 제공합니다:")
    print("1. 사진 평가가 필요 하시면 ('평가해줘'라고 입력하세요)")
    print("2. 사진 비교가 필요 하시면 ('비교해줘'라고 입력하세요")
    print("3. 특정 지역 사진 검색 해줘 ('지역검색'라고 입력하세요")
    print("4. 주변 사진 검색 해줘 ('주변검색'라고 입력하세요")
    print("4. 프로그램 종료 ('종료 라고 입력 하시면 종료 됩니다.')")

    while True:
        user_input = input("\n무엇을 도와드릴까요? ").strip()

        if user_input.lower() in ['종료', 'quit']:
            print("\n PICTO 어플이 만족스러우셨나요? 다음에도 이용 부탁드립니다. 감사합니다🖐️.")
            break

        if "평가해줘" in user_input:
            print(f"\n기본 이미지 경로: {DEFAULT_IMAGE_PATH}")
            image_path = input("분석할 이미지 경로를 입력하세요 (엔터시 기본 이미지 사용): ").strip()

            if not image_path:
                image_path = DEFAULT_IMAGE_PATH

            if not os.path.exists(image_path):
                print("이미지 파일을 찾을 수 없습니다.")
                continue

            print("\n이미지 분석 및 추천 중...")
            result = recommender.process_user_request(user_input, image_path)

            if result.get("상태") == "오류":
                print(f"\n오류: {result['메시지']}")
            elif result.get("상태") == "안내":
                print(f"\n안내: {result['메시지']}")
            else:
                print("\n=== 이미지 분석 결과 ===")
                print(result["이미지 분석"])
                print("\n=== 촬영 가이드라인 ===")
                print(result["촬영 가이드라인"])

        else:
            # GPT와 대화
            print("\nGPT 응답 생성 중...")
            response = chat_with_gpt(recommender, user_input)
            print("\nGPT 응답:")
            print(response)

if __name__ == "__main__":
    main()