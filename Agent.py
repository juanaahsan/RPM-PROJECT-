import cv2
import numpy as np
from PIL import Image

class Agent:
    IMAGE_PIXEL_SIZE = (184, 184)
    
    def __init__(self):
        pass
    
    def Solve(self, problem):
        images = {}
        for figure_name, figure in problem.figures.items():
            img = self.preprocess_image(figure.visualFilename)
            if img is not None:
                images[figure_name] = img
            else:
                return -1

        problem_set_name = problem.problemSetName.lower()
        problem_name = problem.name.lower()

        if "problems d" in problem_set_name or "problems d" in problem_name:
            return self.solve_problem_d(images, problem.problemType)
        elif "problems e" in problem_set_name or "problems e" in problem_name:
            return self.solve_problem_e(images, problem.problemType)
        else:
            if problem.problemType == '2x2':
                return self.solve_2x2(images)
            elif problem.problemType == '3x3':
                return self.solve_3x3(images)
            else:
                return -1

    def preprocess_image(self, file_path):
        try:
            image = Image.open(file_path).convert('L').resize(self.IMAGE_PIXEL_SIZE)
            image_array = np.array(image)
            _, binary_image = cv2.threshold(image_array, 128, 255, cv2.THRESH_BINARY)
            return binary_image
        except:
            return None

    def solve_problem_d(self, images, problem_type):
        if problem_type == '2x2':
            return self.solve_d_2x2(images)
        elif problem_type == '3x3':
            return self.solve_d_3x3(images)
        else:
            return -1

    def solve_d_2x2(self, images):
        A = images['A']
        B = images['B']
        C = images['C']
        candidates = {key: images[key] for key in images if key.isdigit()}

        best_transformation = self.find_best_transformation(A, B)
        transformed_C = self.apply_transformation(C, best_transformation)

        best_match = -1
        lowest_difference = float('inf')
        for key, D in candidates.items():
            difference = self.compute_ipr(transformed_C, D)
            if difference < lowest_difference:
                lowest_difference = difference
                best_match = int(key)
        return best_match

    def solve_d_3x3(self, images):
        positions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        features = {pos: images[pos] for pos in positions if pos in images}
        candidates = {key: images[key] for key in images if key.isdigit()}

        transformations_row = []
        for pair in [('A', 'B'), ('D', 'E')]:
            t = self.find_best_transformation(features[pair[0]], features[pair[1]])
            transformations_row.append(t)

        transformations_col = []
        for pair in [('A', 'D'), ('B', 'E')]:
            t = self.find_best_transformation(features[pair[0]], features[pair[1]])
            transformations_col.append(t)

        if 'H' in features:
            transformed_H = self.apply_transformation(features['H'], transformations_row[-1])
        else:
            transformed_H = None

        if 'F' in features:
            transformed_F = self.apply_transformation(features['F'], transformations_col[-1])
        else:
            transformed_F = None


        if transformed_H is not None and transformed_F is not None:
            predicted_I = cv2.addWeighted(transformed_H, 0.5, transformed_F, 0.5, 0)
        elif transformed_H is not None:
            predicted_I = transformed_H
        elif transformed_F is not None:
            predicted_I = transformed_F
        else:
            return -1

        best_match = -1
        lowest_difference = float('inf')
        for key, I_candidate in candidates.items():
            difference = self.compute_ipr(predicted_I, I_candidate)
            if difference < lowest_difference:
                lowest_difference = difference
                best_match = int(key)
        return best_match

    def find_best_transformation(self, img1, img2):
        transformations = []
        for angle in [0, 90, 180, 270]:
            rotated_img1 = self.rotate_image(img1, angle)
            diff = self.compute_ipr(rotated_img1, img2)
            transformations.append(('rotate', angle, diff))
        for flip_code in [0, 1, -1]:
            flipped_img1 = cv2.flip(img1, flip_code)
            diff = self.compute_ipr(flipped_img1, img2)
            transformations.append(('flip', flip_code, diff))
        best_transformation = min(transformations, key=lambda x: x[2])
        return best_transformation

    def apply_transformation(self, image, transformation):
        if transformation[0] == 'rotate':
            return self.rotate_image(image, transformation[1])
        elif transformation[0] == 'flip':
            return cv2.flip(image, transformation[1])
        else:
            return image

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return rotated

    def compute_ipr(self, img1, img2):
        intersection = cv2.bitwise_and(img1, img2)
        union = cv2.bitwise_or(img1, img2)
        intersection_pixels = np.count_nonzero(intersection)
        union_pixels = np.count_nonzero(union)
        if union_pixels == 0:
            return float('inf')
        else:
            ratio = 1 - (intersection_pixels / union_pixels)
            return ratio

    def solve_problem_e(self, images, problem_type):
        if problem_type == '2x2':
            return self.solve_e_2x2(images)
        elif problem_type == '3x3':
            return self.solve_e_3x3(images)
        else:
            return -1

    def solve_e_2x2(self, images):
        A = images['A']
        B = images['B']
        C = images['C']
        candidates = {key: images[key] for key in images if key.isdigit()}

        operations = [cv2.bitwise_and, cv2.bitwise_or, cv2.bitwise_xor]
        best_op = None
        lowest_diff = float('inf')
        for op in operations:
            result = op(A, B)
            diff = self.compute_ipr(result, C)
            if diff < lowest_diff:
                lowest_diff = diff
                best_op = op

        transformed_C = best_op(A, B)
        best_match = -1
        lowest_difference = float('inf')
        for key, D in candidates.items():
            difference = self.compute_ipr(transformed_C, D)
            if difference < lowest_difference:
                lowest_difference = difference
                best_match = int(key)
        return best_match

    def solve_e_3x3(self, images):
        positions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        features = {pos: images[pos] for pos in positions if pos in images}
        candidates = {key: images[key] for key in images if key.isdigit()}
        operations = [cv2.bitwise_and, cv2.bitwise_or, cv2.bitwise_xor]

        best_op_row = None
        lowest_diff_row = float('inf')
        for op in operations:
            result = op(features['A'], features['B'])
            diff = self.compute_ipr(result, features['C'])
            if diff < lowest_diff_row:
                lowest_diff_row = diff
                best_op_row = op

        best_op_col = None
        lowest_diff_col = float('inf')
        for op in operations:
            result = op(features['A'], features['D'])
            diff = self.compute_ipr(result, features['G'])
            if diff < lowest_diff_col:
                lowest_diff_col = diff
                best_op_col = op

        result_row = best_op_row(features['G'], features['H'])
        result_col = best_op_col(features['C'], features['F'])
        predicted_I = cv2.bitwise_or(result_row, result_col)

        best_match = -1
        lowest_difference = float('inf')
        for key, I_candidate in candidates.items():
            difference = self.compute_ipr(predicted_I, I_candidate)
            if difference < lowest_difference:
                lowest_difference = difference
                best_match = int(key)
        return best_match

    def solve_2x2(self, images):
        A = images['A']
        B = images['B']
        C = images['C']
        candidates = {key: images[key] for key in images if key.isdigit()}

        transformation_AB = self.compute_dpr(A, B)
        best_match = -1
        lowest_difference = float('inf')
        for key, D in candidates.items():
            transformation_CD = self.compute_dpr(C, D)
            difference = abs(transformation_AB - transformation_CD)
            if difference < lowest_difference:
                lowest_difference = difference
                best_match = int(key)
        return best_match

    def solve_3x3(self, images):
        positions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        features = [images[pos] for pos in positions if pos in images]
        candidates = {key: images[key] for key in images if key.isdigit()}

        transformation_row = []
        for i in range(0, len(features)-2, 3):
            dpr1 = self.compute_dpr(features[i], features[i+1])
            dpr2 = self.compute_dpr(features[i+1], features[i+2])
            transformation_row.append(abs(dpr1 - dpr2))

        transformation_col = []
        for i in range(3):
            if i+6 < len(features):
                dpr1 = self.compute_dpr(features[i], features[i+3])
                dpr2 = self.compute_dpr(features[i+3], features[i+6])
                transformation_col.append(abs(dpr1 - dpr2))
            else:
                dpr1 = self.compute_dpr(features[i], features[i+3])
                transformation_col.append(dpr1)

        best_match = -1
        lowest_difference = float('inf')
        for key, I in candidates.items():
            dpr_HI = self.compute_dpr(features[7], I)
            dpr_GH = self.compute_dpr(features[6], features[7])

            if len(transformation_row) > 2:
                row_diff = abs(transformation_row[2] - abs(dpr_GH - dpr_HI))
            else:
                row_diff = abs(dpr_GH - dpr_HI)

            dpr_CF = self.compute_dpr(features[2], features[5])
            dpr_FI = self.compute_dpr(features[5], I)

            if len(transformation_col) > 2:
                col_diff = abs(transformation_col[2] - abs(dpr_CF - dpr_FI))
            else:
                col_diff = abs(dpr_CF - dpr_FI)

            total_difference = row_diff + col_diff
            if total_difference < lowest_difference:
                lowest_difference = total_difference
                best_match = int(key)
        return best_match

    def compute_dpr(self, img1, img2):
        img1_pixels = np.count_nonzero(img1)
        img2_pixels = np.count_nonzero(img2)
        if img1_pixels + img2_pixels == 0:
            return 0
        else:
            ratio = abs(img1_pixels - img2_pixels) / (img1_pixels + img2_pixels)
            return ratio
