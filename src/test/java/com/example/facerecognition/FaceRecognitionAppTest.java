package com.example.facerecognition;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

class FaceRecognitionAppTest {
    @Test
    void reportsDetectionSummaryForSampleImage() throws Exception {
        FaceRecognitionApp app = new FaceRecognitionApp();
        String report = app.analyze(Path.of("sample-face.png"));

        assertTrue(report.contains("Face detection report"), () -> "Expected a detection report header, but got: " + report);
        assertTrue(report.contains("faces detected:"), () -> "Expected a detected-face count, but got: " + report);
    }
}
