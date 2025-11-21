import easyocr

def main():
    reader = easyocr.Reader(['ko', 'en'], gpu=False)

    image_path = './imgs/AdaptiveThreshold.png'

    results = reader.readtext(
        image_path,
        detail=1,
        paragraph=False
    )

    # bbox 기준 줄 정렬
    lines = []
    for bbox, text, conf in results:
        y = bbox[0][1]
        lines.append((y, text))

    lines.sort(key=lambda x: x[0])

    full_text = "\n".join([x[1] for x in lines])

    print("===== 인식 결과 =====")
    print(full_text)


if __name__ == '__main__':
    main()
