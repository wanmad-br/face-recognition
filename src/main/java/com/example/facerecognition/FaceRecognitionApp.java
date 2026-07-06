package com.example.facerecognition;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class FaceRecognitionApp {
    public static void main(String[] args) throws Exception {
        // Step 1: Validate command-line arguments.
        if (args.length < 2 || "--help".equals(args[0])) {
            printUsage();
            return;
        }

        // Step 2: Parse and validate the two image paths.
        Path imagePath1 = Paths.get(args[0]).toAbsolutePath().normalize();
        Path imagePath2 = Paths.get(args[1]).toAbsolutePath().normalize();

        if (!Files.exists(imagePath1)) {
            throw new IllegalArgumentException("Image 1 not found: " + imagePath1);
        }
        if (!Files.exists(imagePath2)) {
            throw new IllegalArgumentException("Image 2 not found: " + imagePath2);
        }

        // Step 3: Create the application instance and run face recognition.
        FaceRecognitionApp app = new FaceRecognitionApp();
        String report = app.compareImages(imagePath1, imagePath2);

        // Step 4: Print the result to the console.
        System.out.println(report);
    }

    /**
     * Compare faces in two images and determine if the same person appears in both.
     */
    public String compareImages(Path imagePath1, Path imagePath2) throws IOException {
        // Step 1: Validate that both image files exist.
        if (!Files.exists(imagePath1)) {
            throw new IOException("Image not found: " + imagePath1);
        }
        if (!Files.exists(imagePath2)) {
            throw new IOException("Image not found: " + imagePath2);
        }

        // Step 2: Load the native OpenCV library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Step 3: Read both images from disk into OpenCV Mat objects.
        Mat image1 = Imgcodecs.imread(imagePath1.toString());
        Mat image2 = Imgcodecs.imread(imagePath2.toString());

        // Step 4: Validate that both images were loaded successfully.
        if (image1.empty()) {
            throw new IOException("Unable to read image 1: " + imagePath1);
        }
        if (image2.empty()) {
            throw new IOException("Unable to read image 2: " + imagePath2);
        }

        // Step 5: Locate and load the Haar cascade face detector.
        String cascadePath = findCascadePath();
        CascadeClassifier detector = new CascadeClassifier(cascadePath);
        if (detector.empty()) {
            throw new IOException("Unable to load face detector cascade: " + cascadePath);
        }

        // Step 6: Detect faces in both images.
        MatOfRect faces1 = new MatOfRect();
        MatOfRect faces2 = new MatOfRect();
        detector.detectMultiScale(image1, faces1);
        detector.detectMultiScale(image2, faces2);

        List<Rect> faceRects1 = new ArrayList<>(faces1.toList());
        List<Rect> faceRects2 = new ArrayList<>(faces2.toList());

        // Step 7: Build the comparison report.
        StringBuilder builder = new StringBuilder();
        builder.append("Face Recognition Report\n\n");
        builder.append("Image 1: ").append(imagePath1).append("\n");
        builder.append("  - width: ").append(image1.width()).append(", height: ").append(image1.height()).append("\n");
        builder.append("  - faces detected: ").append(faceRects1.size()).append("\n\n");

        builder.append("Image 2: ").append(imagePath2).append("\n");
        builder.append("  - width: ").append(image2.width()).append(", height: ").append(image2.height()).append("\n");
        builder.append("  - faces detected: ").append(faceRects2.size()).append("\n\n");

        // Step 8: Compare faces between the two images.
        if (faceRects1.isEmpty() || faceRects2.isEmpty()) {
            builder.append("Face Comparison Results:\n");
            builder.append("  - No match possible: at least one image has no detected faces\n");
        } else {
            builder.append("Face Comparison Results:\n");
            // For each face in image1, compare with each face in image2.
            for (int i = 0; i < faceRects1.size(); i++) {
                Rect rect1 = faceRects1.get(i);
                Mat face1 = new Mat(image1, rect1);

                double maxSimilarity = 0.0;
                int bestMatchIdx = -1;

                for (int j = 0; j < faceRects2.size(); j++) {
                    Rect rect2 = faceRects2.get(j);
                    Mat face2 = new Mat(image2, rect2);

                    // Step 9: Compare face regions using histogram similarity.
                    double similarity = compareFaceHistograms(face1, face2);

                    if (similarity > maxSimilarity) {
                        maxSimilarity = similarity;
                        bestMatchIdx = j;
                    }

                    face2.release();
                }

                // Step 10: Report the best match for this face.
                builder.append("  - Face 1.").append(i + 1).append(" (Image 1):\n");
                if (bestMatchIdx >= 0) {
                    builder.append("      Best match: Face 2.").append(bestMatchIdx + 1).append(" (Image 2)\n");
                    builder.append("      Similarity score: ").append(String.format("%.2f%%", maxSimilarity * 100)).append("\n");
                    if (maxSimilarity >= 0.75) {
                        builder.append("      Result: MATCH (likely same person)\n");
                    } else if (maxSimilarity >= 0.50) {
                        builder.append("      Result: PARTIAL MATCH (possible same person)\n");
                    } else {
                        builder.append("      Result: NO MATCH (different persons)\n");
                    }
                } else {
                    builder.append("      Result: NO FACES TO COMPARE\n");
                }
                builder.append("\n");

                face1.release();
            }
        }

        // Step 11: Release all OpenCV resources.
        image1.release();
        image2.release();
        faces1.release();
        faces2.release();

        // Step 12: Return the completed report.
        return builder.toString().trim();
    }

    /**
     * Compare two face regions using histogram similarity.
     * Returns a similarity score between 0.0 and 1.0.
     */
    private double compareFaceHistograms(Mat face1, Mat face2) {
        try {
            // Convert faces to grayscale for histogram comparison.
            Mat gray1 = new Mat();
            Mat gray2 = new Mat();
            Imgproc.cvtColor(face1, gray1, Imgproc.COLOR_BGR2GRAY);
            Imgproc.cvtColor(face2, gray2, Imgproc.COLOR_BGR2GRAY);

            // Resize face2 to match face1's dimensions for comparison.
            Mat face2Resized = new Mat();
            Imgproc.resize(gray2, face2Resized, gray1.size());

            // Compute histograms for both faces.
            Mat hist1 = new Mat();
            Mat hist2 = new Mat();
            MatOfInt channels1 = new MatOfInt(0);
            MatOfInt channels2 = new MatOfInt(0);
            MatOfInt histSize = new MatOfInt(256);
            MatOfFloat ranges = new MatOfFloat(0, 256);
            Imgproc.calcHist(List.of(gray1), channels1, new Mat(), hist1, histSize, ranges);
            Imgproc.calcHist(List.of(face2Resized), channels2, new Mat(), hist2, histSize, ranges);

            // Normalize histograms.
            Core.normalize(hist1, hist1, 1, 0, Core.NORM_L2);
            Core.normalize(hist2, hist2, 1, 0, Core.NORM_L2);

            // Compare histograms using correlation method (returns -1 to 1, normalized to 0-1).
            double similarity = Imgproc.compareHist(hist1, hist2, Imgproc.CV_COMP_CORREL);
            similarity = Math.max(0, similarity); // Ensure non-negative

            // Clean up temporary matrices.
            gray1.release();
            gray2.release();
            face2Resized.release();
            hist1.release();
            hist2.release();

            return similarity;
        } catch (Exception e) {
            System.err.println("Error comparing face histograms: " + e.getMessage());
            return 0.0;
        }
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
        // Explain how the user should invoke the application with two images.
        System.out.println("Usage: mvn exec:java -Dexec.args='<path-to-image1> <path-to-image2>'");
        System.out.println("Example: mvn exec:java -Dexec.args='foto_1.png foto_2.png'");
        System.out.println("\nThe app will detect faces in both images and compare them.");
        System.out.println("It reports which faces from image1 match faces in image2.");
    }
}
