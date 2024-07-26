import cv2
import os
import numpy as np
import glob

def capture_images(folder_name='calibration_images', image_prefix='calibration_img', max_images=30, camera_index=0):
   
  
    # Create a folder to store calibration images if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Initialize camera capture
    cap = cv2.VideoCapture(camera_index)  # 0 for default camera, change if you have multiple cameras

    # Variables to count images
    image_count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('Camera Feed - Press "c" to Capture', frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF

        # If 'c' is pressed, save the image
        if key == ord('c'):
            image_count += 1
            image_name = f'{image_prefix}{image_count}.jpg'
            image_path = os.path.join(folder_name, image_name)
            cv2.imwrite(image_path, frame)
            print(f'Saved {image_name}')

        # Break the loop when required number of images is captured
        if image_count >= max_images:
            break

        # Break the loop if 'q' is pressed
        if key == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    print(f'{image_count} images captured and saved to {folder_name} folder.')


def camera_calibration(image_folder='calibration_images', image_format='jpg', output_file='finalcalibration_results.npz'):
    squaresX=7
    squaresY=5 
    squareLength=0.03
    markerLength=0.015
    # Define the ChArUco board parameters and create the board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, aruco_dict)

    # Arrays to store the detected corners and ids
    all_corners = []
    all_ids = []
    image_sizes = []

    # Read calibration images
    images = glob.glob(os.path.join(image_folder, f'*.{image_format}'))

    # Debugging: Print the number of images found
    print(f"Found {len(images)} images for calibration.")

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Error loading image {fname}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers in the image
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        
        if ids is not None:
            # Refine the detection and detect the Charuco corners
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ret > 4:  # Ensure at least 4 corners are detected
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                image_sizes.append(gray.shape[::-1])

                # Draw and display the corners (optional)
                img = cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
                cv2.imshow('img', img)
                cv2.waitKey(500)
            else:
                print(f"Not enough corners detected in image {fname}")
        else:
            print(f"No markers detected in image {fname}")

    cv2.destroyAllWindows()

    # Calibrate the camera using the detected corners and ids
    if len(all_corners) > 0:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=board,
            imageSize=image_sizes[0],  # Use the size of the first image
            cameraMatrix=None,
            distCoeffs=None
        )

        # Print calibration results
        if ret:
            print("Calibration was successful")
            print("Camera matrix:\n", camera_matrix)
            print("Distortion coefficients:\n", dist_coeffs)

            # Save the calibration results
            np.savez(output_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
            print(f"Calibration results saved to {output_file}.")

            return True
        else:
            print("Calibration failed. Please check your images and try again.")
            return False
    else:
        print("Not enough corners for calibration. Please check your images and try again.")
        return False


def video_calibration(calibration_file='finalcalibration_results.npz', camera_index=0):
    # Load calibration data (camera_matrix and dist_coeffs) from the calibration_results.npz file
    calibration_data = np.load(calibration_file)
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['dist_coeffs']

    # Open the camera (use 0 for the default camera or the appropriate index for other cameras)
    cap = cv2.VideoCapture(camera_index)  # Use 0 for the default camera

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Get the width and height of the camera frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Undistort the frame
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_matrix)

        # Display the original and undistorted frames (optional)
        cv2.imshow('Original Frame', frame)
      

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
