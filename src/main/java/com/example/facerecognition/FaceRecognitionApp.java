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
        // Step 1: Check whether the user provided an image path.
        if (args.length == 0 || "--help".equals(args[0])) {
            printUsage();
            return;
        }

        // Step 2: Convert the provided path into an absolute, normalized path.
        Path imagePath = Paths.get(args[0]).toAbsolutePath().normalize();

        // Step 3: Verify that the file exists before trying to process it.
        if (!Files.exists(imagePath)) {
            throw new IllegalArgumentException("Image not found: " + imagePath);
        }

        // Step 4: Create the application instance and run the analysis workflow.
        FaceRecognitionApp app = new FaceRecognitionApp();
        String report = app.analyze(imagePath);

        // Step 5: Print the result to the console.
        System.out.println(report);
    }

    public String analyze(Path imagePath) throws IOException {
        // Step 1: Confirm that the image file still exists.
        if (!Files.exists(imagePath)) {
            throw new IOException("Image not found: " + imagePath);
        }

        // Step 2: Load the native OpenCV library so the Java bindings can work.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Step 3: Read the image from disk into an OpenCV Mat object.
        Mat image = Imgcodecs.imread(imagePath.toString());

        // Step 4: Make sure the image was loaded successfully.
        if (image.empty()) {
            throw new IOException("Unable to read image: " + imagePath);
        }

        // Step 5: Locate the Haar cascade XML file used for face detection.
        String cascadePath = findCascadePath();

        // Step 6: Create a face detector using that cascade file.
        CascadeClassifier detector = new CascadeClassifier(cascadePath);

        // Step 7: Ensure the classifier was loaded correctly.
        if (detector.empty()) {
            throw new IOException("Unable to load face detector cascade: " + cascadePath);
        }

        // Step 8: Run the detector over the image and collect all detected face rectangles.
        MatOfRect faces = new MatOfRect();
        detector.detectMultiScale(image, faces);

        // Step 9: Convert the OpenCV rectangles into a Java-friendly list.
        List<Rect> faceRects = new ArrayList<>(faces.toList());

        // Step 10: Build a readable report with image details and detected faces.
        StringBuilder builder = new StringBuilder();
        builder.append("Face detection report\n")
                .append("- image: ").append(imagePath).append("\n")
                .append("- width: ").append(image.width()).append("\n")
                .append("- height: ").append(image.height()).append("\n")
                .append("- faces detected: ").append(faceRects.size()).append("\n");

        // Step 11: Add one line for each detected face with its bounding box coordinates.
        for (int i = 0; i < faceRects.size(); i++) {
            Rect rect = faceRects.get(i);
            builder.append("- face ").append(i + 1)
                    .append(": x=").append(rect.x)
                    .append(", y=").append(rect.y)
                    .append(", w=").append(rect.width)
                    .append(", h=").append(rect.height)
                    .append("\n");
        }

        // Step 12: Release OpenCV resources to avoid memory leaks.
        image.release();
        faces.release();

        // Step 13: Return the completed report.
        return builder.toString().trim();
    }

    private static String findCascadePath() {
        // Check common installation locations for the OpenCV Haar cascade XML file.
        String[] candidates = {
                "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/local/share/opencv/haarcascades/haarcascades/haarcascade_frontalface_default.xml"
        };

        // Try each possible path until the cascade file is found.
        for (String path : candidates) {
            if (Files.exists(Path.of(path))) {
                return path;
            }
        }

        // If none of the expected locations contain the cascade, fail clearly.
        throw new IllegalStateException("OpenCV Haar cascade not found. Install opencv or provide a custom cascade path.");
    }

    private static void printUsage() {
        // Explain how the user should invoke the application from the command line.
        System.out.println("Usage: mvn exec:java -Dexec.args='<path-to-image>'");
        System.out.println("Example: mvn exec:java -Dexec.args='sample-face.png'");
    }
}
