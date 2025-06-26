import cv2
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


class PeopleDetector:
    """
    Класс для детекции людей на видео с использованием YOLOv8.

    Attributes:
        model (YOLO): Загруженная модель YOLOv8
        classes (list): Список классов для детекции (только люди)
    """

    def __init__(self, model_path="yolov8n.pt"):
        """
        Инициализация детектора с загрузкой модели YOLOv8.

        Args:
            model_path (str): Путь к файлу весов YOLO или название модели
        """
        self.model = YOLO(model_path)
        self.classes = [0]  #класс person в COCO

    def detect_people(self, frame, confidence_threshold=0.5):
        """
        Детекция людей на кадре.

        Args:
            frame (numpy.ndarray): Входной кадр видео
            confidence_threshold (float): Порог уверенности для детекции

        Returns:
            list: Список обнаруженных людей в формате (x, y, w, h, confidence)
        """
        results = self.model(frame, verbose=False, classes=self.classes, conf=confidence_threshold)

        people = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                people.append((x1, y1, x2 - x1, y2 - y1, confidence))

        return people

    def draw_detections(self, frame, detections):
        """
        Отрисовка детекций на кадре.

        Args:
            frame (numpy.ndarray): Входной кадр
            detections (list): Список детекций в формате (x, y, w, h, confidence)

        Returns:
            numpy.ndarray: Кадр с отрисованными детекциями
        """
        for (x, y, w, h, confidence) in detections:
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f"person: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


def process_video(input_path="crowd.mp4", output_dir="output_videos", model_size="n", show_progress=True):
    """
    Обработка видеофайла с детекцией людей и сохранение результата.

    Args:
        input_path (str): Путь к входному видеофайлу
        output_dir (str): Директория для сохранения результата
        model_size (str): Размер модели YOLOv8 (n, s, m, l, x)
        show_progress (bool): Показывать прогресс-бар
    """
    Path(output_dir).mkdir(exist_ok=True)

    detector = PeopleDetector(f"yolov8{model_size}.pt")

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(output_dir) / f"processed_{Path(input_path).name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames, desc="Processing video", disable=not show_progress)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect_people(frame)

        processed_frame = detector.draw_detections(frame.copy(), detections)

        out.write(processed_frame)
        pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pbar.close()
    print(f"Processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    process_video()