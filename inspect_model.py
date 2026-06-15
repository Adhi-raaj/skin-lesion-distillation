from models.auxiliary_modules import RareDiseaseClassifier

model = RareDiseaseClassifier(
    backbone="mobilenetv2_100",
    num_classes=7,
    pretrained=False,
    dropout=0.4
)

print(model)