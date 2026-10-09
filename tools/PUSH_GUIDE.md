# GitHub 푸시 안내 (직접 푸시용)

이 저장소의 원격 주소는 다음과 같습니다.

https://github.com/1130dlwltn2-sondol/rag-finetuning-practice

아래 순서대로 실행하면 로컬 작업물을 GitHub에 올릴 수 있습니다.
GitHub 로그인이 필요합니다. (gh auth login 또는 사용자 이름·토큰)

1. 원격 저장소 연결

git remote add origin https://github.com/1130dlwltn2-sondol/rag-finetuning-practice.git

이미 연결되어 있다면 이 단계는 건너뜁니다.
연결 확인: git remote -v

2. 브랜치 이름 확인

git branch -M main

3. 푸시

git push -u origin main

이미 한 번 푸시한 뒤에는 git push 만으로 됩니다.

4. 푸시 확인

브라우저에서 https://github.com/1130dlwltn2-sondol/rag-finetuning-practice 를 열어
README.md, TOC.md, pages/ 24개 파일, assets/ 9개 이미지가 있는지 확인합니다.

주의: API 키, 토큰, 개인정보가 담긴 파일은 절대 커밋하지 마세요.
