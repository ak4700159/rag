import os
import base64
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from config import (
    OPENAI_API_KEY,
    PDF_PATH,
    GPT_MODEL,
    CHROMA_PERSIST_DIR,
    MAX_TOKENS,
    TEMPERATURE
)

class LangchainRecommender:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.current_image_path = None  # 현재 분석 중인 이미지 경로 저장

    def process_pdf(self, pdf_path):
        """PDF 파일을 처리하고 벡터 저장소를 생성합니다."""
        try:
            # 기존 벡터 스토어가 있는지 확인
            if os.path.exists(CHROMA_PERSIST_DIR):
                print("기존 ChromaDB를 불러옵니다.")
                self.vectorstore = Chroma(
                    persist_directory=str(CHROMA_PERSIST_DIR),
                    embedding_function=self.embeddings
                )
                return True

            # 새로운 벡터 스토어 생성
            print("새로운 ChromaDB를 생성합니다.")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            self.vectorstore = Chroma.from_documents(
                documents=pages,
                embedding=self.embeddings,
                persist_directory=str(CHROMA_PERSIST_DIR)
            )
            print("ChromaDB 생성이 완료되었습니다.")
            return True
        except Exception as e:
            print(f"PDF 처리 중 오류 발생: {str(e)}")
            return False

    def add_new_pdf(self, pdf_path):
        """새로운 PDF를 기존 벡터 DB에 추가합니다."""
        try:
            if self.vectorstore is None:
                print("벡터 스토어가 초기화되지 않았습니다. process_pdf()를 먼저 실행해주세요.")
                return False

            print(f"새로운 PDF 추가 중: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            # 기존 벡터 스토어에 새 문서 추가
            self.vectorstore.add_documents(pages)
            print(f"PDF가 성공적으로 추가되었습니다: {pdf_path}")
            return True

        except Exception as e:
            print(f"PDF 추가 중 오류 발생: {str(e)}")
            return False

    def reset_vectorstore(self):
        """벡터 스토어를 초기화합니다."""
        try:
            if os.path.exists(CHROMA_PERSIST_DIR):
                shutil.rmtree(CHROMA_PERSIST_DIR)
                print("벡터 스토어가 초기화되었습니다.")
                self.vectorstore = None
                return True
        except Exception as e:
            print(f"벡터 스토어 초기화 중 오류 발생: {str(e)}")
            return False

    def analyze_image_and_recommend(self, user_input, image_path=None):
        try:
            # 벡터 스토어가 초기화되지 않은 경우
            if self.vectorstore is None:
                print("벡터 스토어가 초기화되지 않았습니다. PDF를 먼저 처리해주세요.")
                return {
                    "이미지 분석": "벡터 스토어 초기화 필요",
                    "촬영 가이드라인": "PDF를 먼저 처리해주세요."
                }

            # 이미지 경로가 제공되지 않은 경우
            if not image_path:
                return {
                    "이미지 분석": "이미지 경로가 제공되지 않았습니다.",
                    "촬영 가이드라인": "이미지 경로를 제공해주세요."
                }

            # 이미지 파일 열기
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # GPT-4 Vision을 사용하여 이미지 분석
            response = self.client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 사진 분석 전문가입니다. 사진을 분석하고 개선을 위한 실용적인 조언을 제공해주세요."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"사용자 요청: {user_input}\n이 사진의 특징과 어떤 색을 가지고 있고 구체적으로 설명해줘, 더 좋은 사진을 위한 조언을 해주세요."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )

            image_analysis = response.choices[0].message.content

            # 분석 결과를 기반으로 가이드라인 검색
            query = f"다음과 같은 사진에 대한 촬영 가이드라인을 제공해주세요: {image_analysis}"
            search_results = self.vectorstore.similarity_search(query)

            recommendations = []
            for doc in search_results:
                recommendations.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'Unknown')
                })

            formatted_results = "\n=== 벡터 DB 검색 결과 ===\n"
            for rec in recommendations:
                formatted_results += f"\n출처: {rec['source']} (페이지: {rec['page']})\n"
                formatted_results += f"{rec['content']}\n"

            return {
                "이미지 분석": image_analysis,
                "촬영 가이드라인": formatted_results
            }

        except Exception as e:
            return {
                "이미지 분석": f"이미지 분석 중 오류 발생: {str(e)}",
                "촬영 가이드라인": "분석 실패로 인해 가이드라인을 제공할 수 없습니다."
            }

    def process_user_request(self, user_input, image_path=None):
        """사용자 요청을 처리하는 메인 메서드"""
        if "평가해줘" in user_input:
            if not image_path:
                return {
                    "상태": "오류",
                    "메시지": "이미지 경로가 제공되지 않았습니다. 이미지 경로를 함께 제공해주세요."
                }
            return self.analyze_image_and_recommend(user_input, image_path)
        else:
            return {
                "상태": "안내",
                "메시지": "이미지 분석을 원하시면 '평가해줘'라는 단어를 포함하여 요청해주세요."
            }

    def check_vectorstore_contents(self):
        """벡터 스토어에 저장된 문서들의 정보를 확인합니다."""
        try:
            if self.vectorstore is None:
                print("벡터 스토어가 초기화되지 않았습니다. PDF를 먼저 처리해주세요.")
                return None

            # 모든 문서 가져오기
            results = self.vectorstore.get()

            if not results['ids']:
                print("벡터 스토어에 저장된 문서가 없습니다.")
                return None

            print("\n=== 벡터 스토어 내용 ===")
            print(f"총 문서 수: {len(results['ids'])}")

            # 각 문서의 메타데이터 확인
            unique_sources = set()
            for metadata in results['metadatas']:
                if 'source' in metadata:
                    unique_sources.add(metadata['source'])

            print("\n저장된 PDF 파일들:")
            for source in unique_sources:
                print(f"- {source}")

            return results

        except Exception as e:
            print(f"벡터 스토어 확인 중 오류 발생: {str(e)}")
            return None

    def analyze_image(self, image_data) :
                # 이미지 파일 열기
        with open(image_data, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # GPT-4 Vision을 사용하여 이미지 분석
        response = self.client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 사진 분석 전문가입니다. 사진을 분석하고 개선을 위한 실용적인 조언을 제공해주세요."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "사진의 구도, 피사체 배치, 사진의 분위기, 색의 조합이 어떠한지 구체적으로 200자 이상으로 설명해줘."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        return response.choices[0].message.content