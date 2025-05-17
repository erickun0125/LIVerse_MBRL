from PIL import Image

# 1. 이미지 파일 경로 지정
image_path = 'subtask/step1.png'  # ← 여기에 PNG 경로 입력

# 2. 이미지 열기
img = Image.open(image_path).convert('RGB')

# 3. 리사이즈
resized_img = img.resize((224, 224))

# 4. 결과 저장 (선택)
resized_img.save('subtask/resized_image.png')

# 5. NumPy 배열로 변환 (선택)
import numpy as np
img_array = np.array(resized_img)
print(img_array.shape)  # (224, 224, 3) 또는 (224, 224, 4) ← 원래 이미지에 알파 채널이 있으면 4채널
