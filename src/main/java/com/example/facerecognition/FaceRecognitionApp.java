package com.example.facerecognition;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class FaceRecognitionApp {
    public static void main(String[] args) throws Exception {
        if (args.length == 0 || "--help".equals(args[0])) {
            printUsage();
            return;
        }

        Path imagePath = Paths.get(args[0]).toAbsolutePath().normalize();
        if (!Files.exists(imagePath)) {
            throw new IllegalArgumentException("Image not found: " + imagePath);
        }

        FaceRecognitionApp app = new FaceRecognitionApp();
        String report = app.analyze(imagePath);
        System.out.println(report);
    }

    public String analyze(Path imagePath) throws IOException {
        if (!Files.exists(imagePath)) {
            throw new IOException("Image not found: " + imagePath);
        }

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat image = Imgcodecs.imread(imagePath.toString());
        if (image.empty()) {
            throw new IOException("Unable to read image: " + imagePath);
        }

        String cascadePath = findCascadePath();
        CascadeClassifier detector = new CascadeClassifier(cascadePath);
        if (detector.empty()) {
            throw new IOException("Unable to load face detector cascade: " + cascadePath);
        }

        MatOfRect faces = new MatOfRect();
        detector.detectMultiScale(image, faces);

        List<Rect> faceRects = new ArrayList<>(faces.toList());
        StringBuilder builder = new StringBuilder();
        builder.append("Face detection report\n")
                .append("- image: ").append(imagePath).append("\n")
                .append("- width: ").append(image.width()).append("\n")
                .append("- height: ").append(image.height()).append("\n")
                .append("- faces detected: ").append(faceRects.size()).append("\n");

        for (int i = 0; i < faceRects.size(); i++) {
            Rect rect = faceRects.get(i);
            builder.append("- face ").append(i + 1)
                    .append(": x=").append(rect.x)
                    .append(", y=").append(rect.y)
                    .append(", w=").append(rect.width)
                    .append(", h=").append(rect.height)
                    .append("\n");
        }

        image.release();
        faces.release();
        return builder.toString().trim();
    }

    private static String findCascadePath() {
        String[] candidates = {
                "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
        };

        for (String path : candidates) {
            if (Files.exists(Path.of(path))) {
                return path;
            }
        }

        throw new IllegalStateException("OpenCV Haar cascade not found. Install opencv or provide a custom cascade path.");
    }

    private static void printUsage() {
        System.out.println("Usage: mvn exec:java -Dexec.args='<path-to-image>'");
        System.out.println("Example: mvn exec:java -Dexec.args='sample-face.png'");
    }
}
