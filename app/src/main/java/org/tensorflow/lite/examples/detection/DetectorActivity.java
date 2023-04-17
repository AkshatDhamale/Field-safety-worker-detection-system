/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.media.MediaPlayer;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.CheckBox;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.DetectorFactory;
import org.tensorflow.lite.examples.detection.tflite.YoloV5Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    public static MediaPlayer player = new MediaPlayer();

    public static AssetManager globalassetmanager;

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.3f;
    private static final boolean MAINTAIN_ASPECT = true;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 640);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private YoloV5Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    private CheckBox helmet, mask, gloves, vest, shoes;

    public int HELMET = 0, MASK = 0, VEST = 0, SHOES = 0, GLOVES = 0;



    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        final int modelIndex = modelView.getCheckedItemPosition();
        final String modelString = modelStrings.get(modelIndex);

        try {
            detector = DetectorFactory.getDetector(getAssets(), modelString);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        int cropSize = detector.getInputSize();

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    public  ArrayList<Object> getValues(){
        ArrayList<Object> vals = new ArrayList<Object>();

        vals.add(this.HELMET);
        vals.add(this.MASK);
        vals.add(this.GLOVES);
        vals.add(this.VEST);
        vals.add(this.SHOES);

        return vals;
    }

    protected void updateActiveModel() {
        // Get UI information before delegating to background
        final int modelIndex = modelView.getCheckedItemPosition();
        final int deviceIndex = deviceView.getCheckedItemPosition();
        String threads = threadsTextView.getText().toString().trim();
        final int numThreads = Integer.parseInt(threads);

        handler.post(() -> {
            if (modelIndex == currentModel && deviceIndex == currentDevice
                    && numThreads == currentNumThreads) {
                return;
            }
            currentModel = modelIndex;
            currentDevice = deviceIndex;
            currentNumThreads = numThreads;

            // Disable classifier while updating
            if (detector != null) {
                detector.close();
                detector = null;
            }

            // Lookup names of parameters.
            String modelString = modelStrings.get(modelIndex);
            String device = deviceStrings.get(deviceIndex);

            LOGGER.i("Changing model to " + modelString + " device " + device);

            // Try to load model.

            try {
                detector = DetectorFactory.getDetector(getAssets(), modelString);
                // Customize the interpreter to the type of device we want to use.
                if (detector == null) {
                    return;
                }
            }
            catch(IOException e) {
                e.printStackTrace();
                LOGGER.e(e, "Exception in updateActiveModel()");
                Toast toast =
                        Toast.makeText(
                                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                toast.show();
                finish();
            }


            if (device.equals("CPU")) {
                detector.useCPU();
            } else if (device.equals("GPU")) {
                detector.useGpu();
            } else if (device.equals("NNAPI")) {
                detector.useNNAPI();
            }
            detector.setNumThreads(numThreads);

            int cropSize = detector.getInputSize();
            croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

            frameToCropTransform =
                    ImageUtils.getTransformationMatrix(
                            previewWidth, previewHeight,
                            cropSize, cropSize,
                            sensorOrientation, MAINTAIN_ASPECT);

            cropToFrameTransform = new Matrix();
            frameToCropTransform.invert(cropToFrameTransform);
        });
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public synchronized void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        Log.e("CHECK", "run: " + results.size());

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

//                        player.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
//                            public void onCompletion(MediaPlayer mp) {
//                                mp.release(); // finish current activity
//                            }
//                        });

                        helmet = findViewById(R.id.checkbox_helmet);
                        mask = findViewById(R.id.checkbox_mask);
                        gloves = findViewById(R.id.checkbox_gloves);
                        vest = findViewById(R.id.checkbox_vest);
                        shoes = findViewById(R.id.checkbox_shoes);


                        for(final Classifier.Recognition result : results){
                            if (result.getTitle().equals("helmet")){
                                helmet.setChecked(true);
                                HELMET = 1;
                            } else if (result.getTitle().equals("Mask")){
                                mask.setChecked(true);
                                MASK = 1;
                            } else if (result.getTitle().equals("Gloves")){
                                gloves.setChecked(true);
                                GLOVES = 1;
                            } else if (result.getTitle().equals("vest")){
                                vest.setChecked(true);
                                VEST = 1;
                            } else if (result.getTitle().equals("safety_shoe")){
                                shoes.setChecked(true);
                                SHOES = 1;
                            }
                        }

                        if (HELMET == 1 && VEST == 1 && MASK == 1 && GLOVES == 1 && SHOES == 1){
                            try {
//                                AssetFileDescriptor afd = getAssets().openFd("verified.mp3");
//                                player.reset();
//                                player.setDataSource(afd.getFileDescriptor(), afd.getStartOffset(), afd.getLength());
//                                afd.close();
//                                player.prepare();
//                                player.start();
                                HELMET = 0;
                                helmet.setChecked(false);
                                VEST = 0;
                                vest.setChecked(false);
                                MASK = 0;
                                mask.setChecked(false);
                                GLOVES = 0;
                                gloves.setChecked(false);
                                SHOES = 0;
                                shoes.setChecked(false);

                            } catch (Exception e) {
                                throw new RuntimeException(e);
                            }
                        }

                        Log.d("HELMET : ", Integer.toString(HELMET));
                        Log.d("VEST : ", Integer.toString(VEST));
                        Log.d("MASK : ", Integer.toString(MASK));
                        Log.d("GLOVES : ", Integer.toString(GLOVES));
                        Log.d("SHOES : ", Integer.toString(SHOES));


                        try {

                            AssetFileDescriptor afd = getAssets().openFd("helmet_mask_vest_gloves_shoes.mp3");
                            System.out.println("YYYYY "+HELMET+" " + MASK + " " + GLOVES + " " + VEST + " " + SHOES);
                            if (HELMET == 0){
                                System.out.println("YYYYY HELMET 0");
                                if (VEST == 0){
                                    if (MASK == 0){
                                        if (GLOVES == 0){
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("helmet_mask_vest_gloves_shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("helmet_mask_vest_gloves.mp3");
                                            }
                                        } else {
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("helmet_mask_vest_shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("helmet mask and vest.mp3");
                                            }
                                        }
                                    } else {
                                        if (GLOVES == 0){
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("helmet vest gloves shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("helmet vest gloves.mp3");
                                            }
                                        } else {
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("helmet vest shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("helmet vest.mp3");
                                            }
                                        }
                                    }
                                    // HELMET == 0 ,VEST == 1
                                } else { // afd = getAssets().openFd("helmet_mask_vest_shoes.mp3");
                                    if (MASK == 0){
                                        if (GLOVES == 0){
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("helmet mask gloves shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("helmet mask gloves.mp3");
                                            }
                                        } else { // GLOVES == 1
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("helmet mask shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("helmet mask.mp3");
                                            }
                                        }
                                    } else { // MASK == 1 HELMET == 0 VEST == 1
                                        if (GLOVES == 0){
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("helmet gloves shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("helmet gloves.mp3");
                                            }
                                        } else { // GLOVES == 1
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("helmet shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("helmet.mp3");
                                            }
                                        }
                                    }
                                }
                                // afd = getAssets().openFd("helmet_mask_vest_shoes.mp3");
                            } else { // HELMET == 1
                                System.out.println("YYYYY HELMET 1");
                                if (VEST == 0){
                                    System.out.println("YYYYY VEST 0");
                                    if (MASK == 0){
                                        if (GLOVES == 0){
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("maks vest gloves shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("mask vest gloves.mp3");
                                            }
                                        } else { // GLOVES == 1
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("mask vest shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("mask vest.mp3");
                                            }
                                        }
                                    } else {
                                        System.out.println("YYYYY MASK 1");// MASK == 1 HELMET == 1 VEST == 0
                                        if (GLOVES == 0){
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("vest gloves shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("vest gloves.mp3");
                                            }
                                        } else {
                                            System.out.println("YYYYY GLOVES 1 MP3");// GLOVES == 1
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("vest shoes.mp3");
                                            } else {
                                                System.out.println("VEST MP3");
                                                afd = getAssets().openFd("vest.mp3");
                                            }
                                        }
                                    }
                                } else { // VEST == 1, HELMET == 1
                                    if (MASK == 0){
                                        if (GLOVES == 0){
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("mask gloves shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("mask gloves.mp3");
                                            }
                                        } else { // GLOVES == 1
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("mask shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("mask.mp3");
                                            }
                                        }
                                    } else { // MASK == 1 HELMET == 1 VEST == 1
                                        if (GLOVES == 0){
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("gloves shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("gloves.mp3");
                                            }
                                        } else { // GLOVES == 1
                                            if (SHOES == 0){
                                                afd = getAssets().openFd("shoes.mp3");
                                            } else {
                                                afd = getAssets().openFd("verified.mp3");
                                            }
                                        }
                                    }
                                }
                            }
                            if (!player.isPlaying()){
                                player.reset();
                                player.setDataSource(afd.getFileDescriptor(), afd.getStartOffset(), afd.getLength());
                                afd.close();
                                player.prepare();
                                player.start();
                            }



                        } catch(Exception e){
                            throw new RuntimeException(e);
                        }


                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {
                                canvas.drawRect(location, paint);

                                cropToFrameTransform.mapRect(location);

                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }
}
