import os
import argparse
import cv2
import numpy as np
import cv2.aruco as aruco
import math
import json

from scipy.spatial.transform import Rotation as R

CALIBRATION_CV2_NAME    = "calibration_cv2.npz"
CALIBRATION_COLMAP_NAME = "cameras.txt"
IMAGES_TMP_NAME         = "images.txt"


def main():
    print("[INFO] Loading images...")
    images, dirs = load_images(args.folder_dir)

    if args.calib:

        print("[INFO] Calibrating camera...")

        if not os.path.exists(args.calibration_path):
            os.mkdir(args.calibration_path)

        calibrate_cv2(images, args.calibration_path)

        print("[INFO] Calibration done.")

        exit(0)

    if not os.path.exists(args.calibration_path):
        print("[ERROR] Calibration data not found")
        exit(1)

    mtx, dist = load_calibration(
        os.path.join(args.calibration_path, CALIBRATION_CV2_NAME)
    )

    poses, directories = process(mtx, dist, images, dirs)

    to_file(poses, directories, args.calibration_path)

    process_script(
        args.folder_dir,
        os.path.join(args.calibration_path, CALIBRATION_COLMAP_NAME),
        os.path.join(args.calibration_path, IMAGES_TMP_NAME),
        args.output
    )

    print("[INFO] Done.")


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""Aruco to NERF script to extract poses using a aruco board 3x3.
                       The separetion between aruco and charuco is 0.01m (default). The size of the aruco is 0.01m (default).
                       If you do not have the camera calibration done, use the --calib attribute."""
    )

    parser.add_argument(
        "--folder_dir",
        type=str,
        default="images",
        help="""Folder containing images to be processed or
                the calibration images if --calib is set""",
    )

    parser.add_argument(
        "--calibration_path",
        type=str,
        default="calibration_data",
        help="""Path to calibration data folder, if argument
                --calib is used this path will be created and
                calibration data will be saved there.""",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="transforms_aruco_copa.json",
        help="""Path to output file the transforms json file."""
    )

    parser.add_argument(
        "--calib",
        action='store_true',
        default=False,
        help="""Calibrate the camera, needs to be run first to generate calibration data.
                The calibration needs images with charuco board DICT_6X6_250.
                The extra info of charuco board is inside of script.""",
    )

    parser.add_argument(
        "--arucosep",
        type=float,
        default=0.01,
        help="""Separation between arucos in m on the aruco board or charuco board."""
    )

    parser.add_argument(
        "--arucosize",
        type=float,
        default=0.01,
        help="""Size of each aruco in m."""
    )

    parser.add_argument(
        "--nrows",
        type=int,
        default=3,
        help="""Number of rows of the aruco board or charuco board."""
    )

    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="""Number of columns of the aruco board or charuco board."""
    )

    parser.add_argument(
        "--dict",
        type=str,
        default="DICT_6X6_250",
        help="""Dictionary of the aruco board or charuco board."""
    )

    return parser.parse_args()


def to_file(transforms, directories, calibration_path):
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    output = ""
    for transform, directory in zip(transforms, directories):
        q = transform[0]
        t = transform[1]
        output += (
            f"0 {q[3]} {-q[0]} {q[1]} {-q[2]} {t[0]} {t[1]} {t[2]} 0 {directory} \n"
        )
        output += "NONE\n"

    out_dir = os.path.join(calibration_path, IMAGES_TMP_NAME)

    if os.path.exists(out_dir):
        os.remove(out_dir)

    with open(out_dir, "w") as f:
        f.write(output)


def load_images(folder_dir):
    dirs = [
        img
        for img in os.listdir(folder_dir)
        if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")
    ]
    images = [cv2.imread(os.path.join(folder_dir, img)) for img in dirs]
    dirs = [os.path.join(folder_dir, img) for img in dirs]
    return images, dirs


def load_calibration(calibration_cv2_path):
    print("[INFO] Loading calibration data...")
    with np.load(calibration_cv2_path) as X:
        mtx, dist, _, _ = [X[i] for i in ("mtx", "dist", "rvecs", "tvecs")]
    return mtx, dist


def calibrate_cv2(images, output_path):

    print("[INFO] Init calibration")

    all_corners = []
    all_ids = []

    board = cv2.aruco.CharucoBoard_create(
        CHARUCO_SIZE[0],
        CHARUCO_SIZE[1],
        M_SIZE_CHARUCO_BOARD,
        M_SIZE_CHARUCO_ARUCO,
        aruco.Dictionary_get(DICTIONARY),
    )

    h, w = images[0].shape[:2]
    record_count = 0

    print("[INFO] Processing images")

    for i, image in enumerate(images):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res = aruco.detectMarkers(
            img_gray, aruco.getPredefinedDictionary(DICTIONARY)
        )

        if len(res[0]) == 8:
            res2 = aruco.interpolateCornersCharuco(res[0], res[1], img_gray, board)

            if res2[1] is not None and res2[2] is not None and len(res2[1]) > (CHARUCO_SIZE//2) + 1:
                all_corners.append(res2[1])
                all_ids.append(res2[2])
                record_count += 1
        
        print(f"\t {i+1} images processed of {len(images)}")

    # Check if recordings have been made
    if record_count != 0:

        print(f"[INFO] Calibrating camera with {record_count} recordings")

        ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, img_gray.shape, None, None
        )

        np.savez_compressed(
            os.path.join(output_path, CALIBRATION_CV2_NAME),
            ret=ret,
            mtx=mtx,
            dist=dist,
            rvecs=rvecs,
            tvecs=tvecs,
        )

        print("[INFO] Saving Calibration to numPy file")

        dist = dist.squeeze()

        fx = mtx[0, 0]
        fy = mtx[1, 1]
        cx = mtx[0, 2]
        cy = mtx[1, 2]

        k1 = dist[0]
        k2 = dist[1]

        p1 = dist[2]
        p2 = dist[3]

        values = [fx, fy, cx, cy, k1, k2, p1, p2]

        out = "1 OPENCV "
        out += str(w) + " " + str(h) + " "
        for value in values:
            out += "{:.20f}".format(value) + " "

        out_file = os.path.join(output_path, CALIBRATION_COLMAP_NAME)

        if os.path.exists(out_file):
            os.remove(out_file)

        with open(out_file, "w") as f:
            f.write(out)

        print("[INFO] Saving Calibration txt file")

    else:
        print("Interrupted since there are no records...")


def process(mtx, dist, images, dirs):

    print("[INFO] Calculating transforms from aruco board")

    poses = []
    directories = []

    charuco_board = aruco.GridBoard_create(
        ARUCO_SIZE[0],
        ARUCO_SIZE[1],
        M_SIZE_ARUCO,
        M_SEPARATION_ARUCO,
        aruco.Dictionary_get(DICTIONARY),
    )

    rvec__ = None
    tvec__ = None

    index = 0

    for image, path in zip(images, dirs):

        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        res = aruco.detectMarkers(img_gray, aruco.getPredefinedDictionary(DICTIONARY))

        boleano, rvecs, tvecs = aruco.estimatePoseBoard(
            res[0], res[1], charuco_board, mtx, dist, rvec__, tvec__
        )

        if not boleano:
            continue

        rvec = rvecs.squeeze()
        tvec = tvecs.squeeze()

        rvec[0] = -rvec[0]
        rvec[2] = -rvec[2]
        R_ = cv2.Rodrigues(rvec)[0]
        camQvec = R.from_matrix(R_).as_quat()

        poses.append((camQvec, tvec))
        directories.append(path)
        index += 1

        print(f"\t {index} images processed of {len(images)}")
    
    print("[INFO] POSES done")

    return poses, directories


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def closest_point_2_lines(
    oa, da, ob, db
):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def process_script(images_dir, cameras_file, images_file, output_path, aabb_scale=16):

    print("[INFO] Preparing images to intant ngp format...")

    with open(cameras_file, "r") as f:
        angle_x = math.pi / 2
        for line in f:
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            fl_y = float(els[5])
            cx = float(els[6])
            cy = float(els[7])
            k1 = float(els[8])
            k2 = float(els[9])
            p1 = float(els[10])
            p2 = float(els[11])

            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    with open(images_file, "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aabb_scale": aabb_scale,
            "frames": [],
        }

        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i % 2 == 1:
                elems = line.split(" ")

                name = str(os.path.join(".",f"{'_'.join(elems[9:])}"))
                b = sharpness(name)

                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                c2w[0:3, 2] *= -1  # flip the y and z axis
                c2w[0:3, 1] *= -1
                c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
                c2w[2, :] *= -1  # flip whole world upside down

                up += c2w[0:3, 1]

                frame = {"file_path": name, "sharpness": b, "transform_matrix": c2w}
                out["frames"].append(frame)
    nframes = len(out["frames"])

    up = up / np.linalg.norm(up)

    R = rotmat(up, [0, 0, 1])

    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(
            R, f["transform_matrix"]
        )  # rotate up to be the z axis

    # find a central point they are all looking at
    print("[INFO] computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3, :]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw

    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= totp

    avglen = 0.0
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= nframes

    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    print(f"[INFO] writing {output_path}")
    with open(output_path, "w") as outfile:
        json.dump(out, outfile, indent=2)


if __name__ == "__main__":

    args = get_arguments()

    DICTIONARY = eval(f'aruco.{args.dict}')

    # Constants for calibration board
    M_SIZE_CHARUCO_BOARD = args.arucosep
    M_SIZE_CHARUCO_ARUCO = args.arucosize
    CHARUCO_SIZE         = (args.nrows, args.ncols)


    # Constants for processing images
    M_SEPARATION_ARUCO   = args.arucosep
    M_SIZE_ARUCO         = args.arucosize
    ARUCO_SIZE           = (args.nrows, args.ncols)

    main()
