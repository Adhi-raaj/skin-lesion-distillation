from gradcam_service import GradCAMService

service = GradCAMService()

with open("test.jpg", "rb") as f:
    image_bytes = f.read()

result = service.generate(image_bytes)

print(type(result))
print(len(result))