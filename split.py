import numpy
from scipy.spatial import distance


def split_joint(segment, letters, return_tuple = False):
    '''Split joint letters in segment to multiple segments
    returns (segments, success) tuple, where segments is array of segments, success is a boolean value'''

    def find_top(s, x, nearest_to_y, nearest_to_x):
        '''Find a join point from top to bottom
        s: segment
        x: central point which is near to join point
        nearest_to_ynearest_to_x: the found point will be near to this point as possible'''
        find_nearest = nearest_to_y >= 0 and nearest_to_x >= 0
        nearest_y = -1
        nearest_x = -1
        nearest_dist = float('inf')

        def check_nearest(y, x):
            '''Find a join point from top to bottom
            y, x: point coordinates to calculate a distance'''
            if find_nearest:
                nonlocal nearest_y, nearest_x, nearest_dist
                dist = distance.euclidean((x, y), (nearest_to_x, nearest_to_y))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_y = y
                    nearest_x = x

        def find_internal():
            '''Internal function that performs all magic'''
            nonlocal s, x

            def find_left(y, x):
                '''Find possibility to move bottom moving left
                y, x: start points
                returns a new X point or -1 if there is nowhere to go'''
                while s[y][x-1] != 0 and s[y][x] != 0 and s[y-1][x-1] == 0:
                    check_nearest(y, x)
                    x -= 1
                check_nearest(y, x)
                if s[y][x-1] != 0 and s[y][x] != 0:
                    return -1
                return x

            def find_right(y, x):
                '''Find possibility to move bottom moving right
                y, x: start points
                returns a new X point or -1 if there is nowhere to go'''
                while s[y][x-1] != 0 and s[y][x] != 0 and s[y-1][x] == 0:
                    check_nearest(y, x)
                    x += 1
                check_nearest(y, x)
                if s[y][x-1] != 0 and s[y][x] != 0:
                    return -1
                return x

            first_met_point = None
            for y in range(1, segment.shape[0]):
                if s[y][x-1] != 0 and s[y][x] != 0:
                    if first_met_point is None:
                        first_met_point = (y, x)
                    start_x = x
                    if s[y-1][x-1] == 0 and s[y-1][x] == 0:
                        x1 = find_left(y, x)
                        x2 = find_right(y, x)
                        if x1 == -1 and x2 == -1:
                            return (y, start_x)
                        x = x2 if x2 != -1 and x2 - x >= x - x1 or x1 == -1 else x1
                    elif s[y-1][x-1] == 0 and s[y-1][x] != 0:
                        x = find_left(y, x)
                        if x == -1: return (y, start_x)
                    elif s[y-1][x-1] != 0 and s[y-1][x] == 0:
                        x = find_right(y, x)
                        if x == -1: return (y, start_x)
            return first_met_point if first_met_point is not None else (-1, -1)

        y, x = find_internal()
        if find_nearest and nearest_y >= 0 and nearest_x >= 0:
            return (nearest_y, nearest_x)
        else:
            return (y, x)

    def find_joint(s, center_top, center_bottom):
        '''Find two join points from top and bottom
        s: segment
        center_top: central point which is near to join point from top
        center_bottom: central point which is near to join point from bottom
        returns y1, x1, y1, x2: a tuple with points coordinates, may return -1s for error '''
        height,width = s.shape
        y1,x1 = find_top(s, center_top, -1, -1)
        # find again from bottom with flipping array vertically
        y2,x2 = find_top(numpy.flipud(s), center_bottom, -1, -1)
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0: return (y1, x1, y2, x2)
        y2 = height - y2
        '''assuming that one point is more accurate, find another point again
        which will be closer to accurate point'''
        if abs(x1 - center_top) < abs(x2 - center_bottom):
            y2,x2 = find_top(numpy.flipud(s), center_bottom, height - y1, x1)
            y2 = height - y2
        else:
            y1,x1 = find_top(s, center_top, y2, x2)
        return (y1, x1, y2 + 1, x2)

    def find_preferred_joint(s, center):
        '''Find two join points from top and bottom
        s: segment
        center: central point which is near to join point
        returns y1, x1, y1, x2: a tuple with points coordinates, may return -1s for error '''
        y1, x1, y2, x2 = find_joint(s, center, center)
        if y1 < 0 or x1 < 0 or y2 < 0 or x2 < 0:
            # Try to shift center and find the best option
            nearest_dist = float('inf')
            dlength = 3
            for dtop in [-dlength, 0, dlength]:
                for dbottom in [-dlength, 0, dlength]:
                    if dtop == 0 and dbottom == 0:
                        continue
                    sy1, sx1, sy2, sx2 = find_joint(s, center + dtop, center + dbottom)
                    if sy1 < 0 or sx1 < 0 or sy2 < 0 or sx2 < 0:
                        continue
                    dist = distance.euclidean((sx1, sy1), (sx2, sy2))
                    if dist < nearest_dist:
                        dist,y1,x1,y2,x2 = nearest_dist,sy1,sx1,sy2,sx2
        return None if y1 < 0 or x1 < 0 or y2 < 0 or x2 < 0 else (y1, x1, y2, x2)
        return (y1, x1, y2, x2)

    def create_division(segment, mask, r, x, center, index):
        '''Fill mask with division line
        r: range to iterate vertically
        x: start X point'''
        zero_met = False
        step = 1 if center - 1 > x else -1
        for y in r:
            if segment[y][x] != 0 and zero_met:
                while segment[y][x] != 0: x = x + step
            elif segment[y][x] == 0:
                zero_met = True
            mask[y][x] = index

    def fill_mask(mask, index):
        '''Fill mask array with index until index met
        e.g. tranform 000010000 to 111110000 when index == 1,
        transform 111110000 to 111112222 when index == 2'''
        height, width = mask.shape
        for y in range(0, height):
            for x in range(0, width):
                if mask[y][x] == 0:
                    mask[y][x] = index
                elif mask[y][x] == index:
                    break

    def trim_matrix(matrix):
        '''Remove zero lines and columns'''
        for i in range(0, 2):
            matrix = numpy.transpose(matrix[~(matrix == 0).all(1)])
        return matrix

    def extend_matrix(matrix):
        '''Return extended matrix with row and column around the edges filled with 0'''
        return numpy.pad(matrix, 1, 'constant', constant_values = 0)

    def narrow_matrix(matrix):
        '''Return narrowed matrix with removed row and column around the edges'''
        height, width = matrix.shape
        return matrix[1:height-1,1:width-1]

    segments = list([segment])
    success = True
    mask = None
    if letters >= 2:
        segment = extend_matrix(segment)
        mask = numpy.zeros(segment.shape, numpy.uint8)
        for i in range(1, letters):
            center = segment.shape[1] * i // letters
            find_result = find_preferred_joint(segment, center)
            if find_result is None:
                success = False
                mask = None
                break;
            y1, x1, y2, x2 = find_result
            create_division(segment, mask, range(y1, 0, -1), x1, center, i)
            create_division(segment, mask, range(y2, segment.shape[0]), x2, center, i)
            # Unite two division lines
            for y in range(y1 + 1, y2):
                mask[y][x1 + (x2 - x1) * (y - y1) // (y2 - y1)] = i
            fill_mask(mask, i)
        segment = narrow_matrix(segment)
        if success:
            fill_mask(mask, letters)
            mask = narrow_matrix(mask)
            segments.clear()
            for i in range(1, letters + 1):
                cs = numpy.copy(segment)
                # Apply mask
                height, width = mask.shape
                for y in range(0, height):
                    for x in range(0, width):
                        if mask[y][x] != i:
                            cs[y][x] = 0
                segments.append(trim_matrix(cs))
    return (segments, success) if return_tuple else segments
