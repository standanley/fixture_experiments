import numpy as np
import scipy
np.random.seed(4)
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.optimize
import itertools
import time
from collections import defaultdict
#from fixture import Sampler

#import cProfile

def get_data():
    # two planes
    x = np.random.random(1000)
    y = np.random.random(1000)
    z_low = x*0.2 + y*0.4 + 0.3
    #z_high = x*1.2 + y*1.6 - 0.9
    z_high = x*2.2 + y*3.6 - 2.9
    z = np.max((z_low, z_high), axis=0) + np.random.normal(0, 0.01, z_low.shape)

    print('test err', np.linalg.norm(np.max((z_low, z_high), axis=0)-z))
    # intersection line before scaling is y=-5/8x+1, after scaling
    # y=-5/8x+5
    #plot(x, y, z)
    return x, y, z

def get_data2():
    # small ears
    x = np.random.random(1000)
    y = np.random.random(1000)

    z_mid = 0*x + 0*y + 0 #x*0.1 + y*0.2 - 0.15
    z_low = x*3 + y*5 - 0.8
    z_high = x*4 + y*4 -7.2

    z = np.max((np.min((z_low, z_mid), axis=0), z_high), axis=0) + np.random.normal(0, 0.0001, z_low.shape)

    #plot(x, y, z)
    return x, y, z
    #return x, y, z

def get_data3():
    # bowl
    x = np.random.random(1000)
    y = np.random.random(1000)

    z = 1.3*(x-0.2)**2 + 2.2*(y-0.5)**2 - 2

    #plot(x, y, z)
    return x, y, z
    #return x, y, z

def data_transform_4(x, y):
    extra_gain = 1+(y-1/1.3)*1.3
    z = np.tanh((x-.5)*2*extra_gain)
    return z

def get_data4():
    # amp
    x = np.random.random(1000)
    y = np.random.random(1000)

    ##extra_gain = 1 + (y-0.5)*0.6
    #extra_gain = 1+(y-1/1.3)*1.3
    #z = np.tanh((x-.5)*2*extra_gain)
    ##z = (x - .5) * 2 * extra_gain
    z = data_transform_4(x, y)

    #plot(x, y, z)

    #slope_removal = (x-.5)*1.7
    #z = z - slope_removal

    plot(x, y, z)
    #plt.show()
    return x, y, z



def get_data5():
    # twist
    x = np.random.random(500)
    y = np.random.random(500)

    z = (x-0.5) * np.clip((y-0.5), -0.15, 0.15)

    #plot(x, y, z)
    #plt.show()
    return x, y, z

def get_data_3d():
    # amp
    x = np.random.random(500)
    y = np.random.random(500)
    z = np.random.random(500)


    # intersection plane should be 1x+2y+3z-3=0
    # high has higher slope, and both pass through (.5, .5, .5, 0)
    w_low = 0.6*x + 0.2*y + 1.2*z - 1
    w_high = 1.6*x + 2.2*y + 4.2*z - 4
    w = np.max((w_low, w_high), axis=0)

    xs = np.stack((x, y, z), axis=1)
    #plot(x, y, z)
    return xs, w

def get_data_binary():
    NUM_BITS = 3
    x = np.random.random((500))
    binary = np.random.randint(0, 2, size=(500, NUM_BITS))
    weights = 2**np.array(range(NUM_BITS))
    binary_value = np.sum(binary*weights, axis=1)
    z_linear = 2.0*(x-0.5) + 0.1*binary_value
    z = np.tanh(z_linear)

    plot(x, binary_value, z)
    #xs = np.hstack((x.reshape((len(x),1)), binary))
    xs = np.hstack((x.reshape((len(x),1)), 1/2**NUM_BITS*binary_value.reshape((len(x), 1))))
    return xs, z


def plot(x, y, z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z)
    #plt.show()

def plot_multiple(things):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for x, y, z in things:
        ax.scatter(x, y, z)
    plt.show()

def linear_regression(xs, z):
    # find C s.t. X*C ~= Z
    xs_with_const = np.hstack((xs, np.ones((xs.shape[0], 1))))
    C = np.linalg.pinv(xs_with_const) @ z
    preds = (xs_with_const @ C)
    return C, preds

def group(xs, z, regs):
    zs_est = None

def fit(x, y, z):
    N = 3
    xs = np.stack((x, y), 1)
    mins = xs.min(0)
    maxs = xs.max(0)
    seeds = np.random.random((N, xs.shape[1]))*(maxs-mins) + mins

    # now sort datapoints according to closest seed
    # xs -> (100, 1, 2)
    # seeds -> (1, 3, 2), leading 1 can be implicit
    # broadcast to (100, 3, 2) when subtracting
    temp = xs.reshape((xs.shape[0], 1, xs.shape[1]))
    errors = ((temp - seeds)**2).sum(axis=2)

    ITERS = 5
    for i in range(ITERS):
        group_ids = errors.argmin(axis=1)
        #groups = []
        to_plot = []
        all_preds = []
        for i in range(N):
            group = xs[group_ids==i, :]
            group_z = z[group_ids==i]
            #groups.append(group)
            C, preds_partial = linear_regression(group, group_z)
            preds = xs @ C[:-1] + C[-1]
            to_plot.append((group[:, 0], group[:, 1], group_z))
            #to_plot.append((xs[:, 0], xs[:, 1], preds))
            to_plot.append((group[:, 0], group[:, 1], preds_partial))

            all_preds.append(preds)
            print(f'Group {i} size {len(group_z)}')
        plot_multiple(to_plot)
        all_preds = np.stack(all_preds, axis=1)
        errors = (all_preds - z.reshape((z.shape[0], 1)))**2

    pass

def fit2(xs, z):
    assert xs.shape[0] == len(z)
    xs = np.append(xs, np.ones((len(z), 1)), 1)

def fit3(xs, z):
    from sklearn.linear_model import LinearRegression
    from lineartree import LinearTreeRegressor
    regr = LinearTreeRegressor(base_estimator=LinearRegression())
    regr.fit(xs, z)
    z_hat = regr.predict(xs)

    plot_multiple([
        #(xs[:,0], xs[:,1], z),
        (xs[:, 0], xs[:, 1], z_hat),
    ])
    print()


def fit4(x, y, z):
    points = [[0,0], [1,1], [0,1], [1,0]]

    def unpack_state(state):
        points = state[:8].reshape((4,2))
        heights = state[8:]
        return points, heights

    state_init = np.array([0,0,1,1,0,1,1,0,0,0,0,0])
    points, heights = unpack_state(state_init)

    def error(state):
        points, heights = unpack_state(state)
        for xx, yy, zz in zip(x, y, z):
            # (xy - points[0]) dot (points[1] - points[0])
            dot = (  (xx - points[0][0]) * (points[1][0] - points[0][0])
                   + (yy - points[0][1]) * (points[1][1] - points[0][1]))
            if dot <= 0:
                tri = points[0], points[1], points[2]
                hs = heights[0], heights[1], heights[2]
            else:
                tri = points[0], points[3], points[1]
                hs = heights[0], heights[3], heights[1]

            t0 = tri[1] - tri[0]
            t1 = tri[2] - tri[0]
            v = np.array([xx-tri[0][0], yy-tri[0][1]])
            coords = np.linalg.inv(np.stack((t0, t1), 1)) @ v
            print()

    error(state_init)
    print()



class FragmentFit:
    #MAX_ERR = 0.5
    MAX_ERR = 0.001

    class Fragment:
        def __init__(self, fit, id, center):
            self.fit = fit
            self.id = id
            self.center = center
            self.points = []
            self.defunct = False

            dists = np.linalg.norm(self.fit.xs - center, axis=1)
            self.ordering = np.argsort(dists)
            self.i = 0

        def assign_init(self):
            # grab the 2(N+1) nearest points, if they are already claimed
            # just take the next nearest instead
            num = 2*(self.fit.N+1)
            while len(self.points) < num:
                j = self.ordering[self.i]
                self.i += 1
                if self.fit.assignment[j] == -1:
                    self.fit.assignment[j] = self.id
                    self.points.append(j)

        #def test(self):
        #    xs = self.fit.xs[self.points]
        #    z = self.fit.z[self.points]
        #    self.fit.pred(xs, z)

        def grow(self):
            if self.defunct:
                return
            while True:
                if self.i == len(self.ordering):
                    return False
                new_index = self.ordering[self.i]
                self.i += 1
                if self.fit.assignment[new_index] == -1:
                    # found an unclaimed point
                    new_xs = self.fit.xs[self.points + [new_index]]
                    new_z = self.fit.z[self.points + [new_index]]
                    err = self.fit.err(new_xs, new_z)
                    if err < self.fit.MAX_ERR:
                        # add it!
                        self.points.append(new_index)
                        self.fit.assignment[new_index] = self.id
                    # added or not, we are done
                    return
                elif self.fit.assignment[new_index] == self.id:
                    # this point already belongs to us
                    # probably acquired it during a merge with another frag
                    continue
                else:
                    # this point belongs to someone else!
                    other_id = self.fit.assignment[new_index]
                    other = self.fit.fragments[other_id]
                    assert other.id == other_id

                    # can we combine with other?
                    points_combined = self.points + other.points
                    xs_combined = self.fit.xs[points_combined]
                    z_combined = self.fit.z[points_combined]
                    err_combined = self.fit.err(xs_combined, z_combined)
                    if err_combined < self.fit.MAX_ERR:
                        print(f'Merging {other.id} into {self.id}')
                        # combine!
                        self.points += other.points
                        for p in other.points:
                            self.fit.assignment[p] = self.id
                        other.points = []
                        other.defunct = True
                        return

                    # if we are still here, we couldn't combine
                    # should we steal this point?
                    my_err_without = self.fit.err2(self.points)
                    my_err_with = self.fit.err2(self.points + [new_index])
                    other_err_without = self.fit.err2([i for i in other.points if i != new_index])
                    other_err_with = self.fit.err2(other.points)

                    # predict this point based on regression without it
                    point = self.fit.xs[new_index]
                    height = self.fit.z[new_index]
                    my_coefs = self.fit.coefs2(self.points)
                    my_pred = np.hstack((point, 0)) @ my_coefs
                    my_err_point = abs(my_pred - height)
                    other_coefs = self.fit.coefs2([i for i in other.points if i != new_index])
                    other_pred = np.hstack((point, 0)) @ other_coefs
                    other_err_point = abs(other_pred - height)

                    #if my_err_with - my_err_without < other_err_with - other_err_without:
                    #if my_err_with < other_err_with:
                    if my_err_point < other_err_point:
                        # added less error for us, so let's steal it
                        print(f'{new_index} stolen from {other.id} to {self.id}')
                        self.points.append(new_index)
                        self.fit.assignment[new_index] = self.id
                        other.points.remove(new_index)
                        return

                    # guess we don't get this point
                    pass


    def data(self, indices):
        xs = self.xs[indices]
        z = self.z[indices]
        return xs, z

    def coefs(self, xs, z):
        xs_with_const = np.hstack((xs, np.ones((xs.shape[0], 1))))
        coefs = np.linalg.pinv(xs_with_const) @ z
        #print(coefs)
        return coefs

    def coefs2(self, indices):
        return self.coefs(*self.data(indices))

    def pred(self, xs, z):
        coefs = self.coefs(xs, z)
        #print(coefs)
        xs_with_const = np.hstack((xs, np.ones((xs.shape[0], 1))))
        z_pred = xs_with_const @ coefs
        return z_pred

    def pred2(self, indices):
        return self.pred(*self.data(indices))

    def err(self, xs, z):
        z_pred = self.pred(xs, z)
        err = z_pred - z
        return np.max(err) #np.linalg.norm(err)

    def err2(self, indices):
        return self.err(*self.data(indices))

    def plot(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(-1, len(self.fragments)):
            indices = np.where(self.assignment==i)
            xs = self.xs[indices]
            z = self.z[indices]
            ax.scatter(xs[:,0], xs[:,1], z)
        plt.show()
        print()

    def __init__(self, xs, z):
        #assert False, 'Some fragments have error above MAX_ERR; must be a bug'
        self.maxes = np.max(xs, axis=0)
        self.mins = np.min(xs, axis=0)
        self.xs = (xs - self.mins) / (self.maxes - self.mins)
        self.z = z
        self.assignment = np.full(self.xs.shape[0], -1)

        # N = number of input dimension
        # M = initial number of fragments
        self.N = xs.shape[1]
        self.M = 25
        centers = np.random.random((self.M, self.N))

        self.fragments = []
        for id, c in enumerate(centers):
            self.fragments.append(self.Fragment(self, id, c))

        for fragment in self.fragments:
            fragment.assign_init()

        for i in range(100):
            self.plot()
            for i in range(25):
                for frag in self.fragments:
                    frag.grow()
            for frag in self.fragments:
                if frag.defunct:
                    continue
                print()
                print(frag.id)
                print(len(frag.points))
                print(self.err2(frag.points))


def get_unit_simplex(dim):
    # return a d-dimensional simplex where every vertex is distance
    # 1 from every other
    assert dim > 0
    if dim == 1:
        return np.array([[-0.5], [0.5]])
    rec = get_unit_simplex(dim - 1)
    base = np.concatenate((rec, np.zeros((dim, 1))), 1)

    dist_to_center = np.linalg.norm(base[0])
    height = (1 - dist_to_center**2)**0.5

    bump = np.zeros(dim)
    bump[-1] = -1 * height / (dim+1)
    base += bump

    new = np.zeros(dim)
    new[-1] = height * (1 - 1/(dim+1))

    ans = np.concatenate((base, new.reshape(1, dim)), 0)
    return ans

class NetFit:
    class BadSplitException(BaseException):
        pass

    def __init__(self, xs, z):
        self.maxes = np.max(xs, axis=0)
        self.mins = np.min(xs, axis=0)
        self.xs = (xs - self.mins) / (self.maxes - self.mins)
        self.z = z
        self.assignment = np.full(self.xs.shape[0], -1)


        # initial voronoi
        NUM_REGIONS = 4
        xs = np.random.rand(NUM_REGIONS)
        ys = np.random.rand(NUM_REGIONS)
        #zs = np.random.rand(NUM_POINTS)
        data = np.stack((xs, ys), 1)

        # add corners
        INPUT_DIM = data.shape[1]
        corners = []
        for coord in itertools.product((-1,1), repeat=INPUT_DIM):
            corners.append([x*3 for x in coord])
        #for i in range(DIM):
        #    corners.append([-3 if i == j else 0 for j in range(DIM)])
        #    corners.append([ 3 if i == j else 0 for j in range(DIM)])
        corners = np.array(corners)
        data = np.concatenate((data, corners), 0)

        v = scipy.spatial.Voronoi(data)

        print(data)
        print(v.vertices)
        print(np.array(v.regions))
        print(np.array(v.ridge_vertices))
        print(v.ridge_points)

        #good_regions = [r for r in v.regions if -1 not in r]
        # good points are all the ones that aren't added around the boundary
        good_points = set(range(NUM_REGIONS))
        # good ridges are all the ones that separate 2 good points
        # good_ridge_list is [((point_a_i, point_b_i), list_of_ridge_i), ...]
        good_ridge_list = [(points, ridge)
                           for points, ridge in v.ridge_dict.items()
                           if all(p in good_points for p in points)]


        def get_ridge_splits(vertices):
            # ridge split is a normal vector with one extra value tacked on
            # do a dot product of ridge_split with some point with a 1 tacked on
            # dot product positive/negative tells you which side of the ridge
            ridge_splits = []
            for points_i, ridge in good_ridge_list:
                # TODO arbitrary dims
                assert len(ridge) == 2
                ridge_vertices = vertices[ridge]
                tangent = ridge_vertices[1] - ridge_vertices[0]
                normal = np.array([tangent[1], -tangent[0]])
                offset = np.dot(ridge_vertices[0], normal)
                split = np.concatenate((normal, [-offset]), 0)
                ridge_splits.append(split)
            return ridge_splits


        ridge_splits_initial = get_ridge_splits(v.vertices)
        point_split_info = []
        for point_i in good_points:
            # don't need region number because we determine which ridges are
            # for this point by looking at the points defining each ridge
            #region_i = v.point_region[point_i]
            ridges = []
            for i, (points_i, ridge) in enumerate(good_ridge_list):
                if point_i in points_i:
                    split = ridge_splits_initial[i]
                    point = v.points[point_i]
                    point_h = np.concatenate((point, [1]), 0)
                    split_val = np.dot(point_h, split)

                    positive = split_val  > 0
                    ridges.append((i, positive))
            point_split_info.append(ridges)

        def check_region(point, ridge_splits):
            sides = []
            point_h = np.concatenate((point, [1]), 0)
            for split in ridge_splits:
                split_val = np.dot(point_h, split)
                sides.append(split_val > 0)

            candidates = []
            for point_i, ridge_info in enumerate(point_split_info):
                for ridge_i, positive in ridge_info:
                    if sides[ridge_i] != positive:
                        # we fell on the wrond side of this ridge
                        # we know this is not point_i
                        break
                else:
                    # we never broke out, so we got the right side for all
                    candidates.append(point_i)

            if len(candidates) != 1:
                raise self.BadSplitException
            return candidates[0]

        # THE NEXT FEW BLOCKS ARE JUST PLOTTING FOR TEST
        test = [[] for _ in range(len(good_points))]
        for point in self.xs:
            region = check_region(point, ridge_splits_initial)
            test[region].append(point)

        #for ps in test:
        #    ps = np.array(ps)
        #    plt.scatter(ps[:,0], ps[:,1])
        #plt.show()

        #scipy.spatial.voronoi_plot_2d(v)
        #plt.show()


        def regress(vertices):
            num_planes = len(good_points)
            NUM_DATAPOINTS = self.xs.shape[0]

            # For a INPUT_DIM=2 example, we are trying to get a result vector:
            # vec = [a0 b0 c0 a1 b1 c1 ...] (but it's a column vector)
            # where the plane for region 0 is z = a0*x + b0*y + c0
            # we we build our matrix m such that z_pred = m @ vec
            m = np.zeros((NUM_DATAPOINTS, (INPUT_DIM+1)*num_planes))
            ridge_splits = get_ridge_splits(vertices)
            for point_i in range(NUM_DATAPOINTS):
                point = self.xs[point_i]
                assert len(point) == INPUT_DIM
                region = check_region(point, ridge_splits)

                for i in range(INPUT_DIM):
                    m[point_i][(INPUT_DIM+1)*region + i] = point[i]
                m[point_i][(INPUT_DIM + 1) * region + INPUT_DIM] = 1





            vec, errs, rank, s = np.linalg.lstsq(m, self.z)

            return vec, m

        def predict(vertices):
            vec, m = regress(vertices)
            return m @ vec

        def error(vertices):
            pred = predict(vertices)
            errs = pred - self.z
            err = np.linalg.norm(errs)
            print('error: ', err)
            return err

        def plot_attempt(pred, vertices):
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter(self.xs[:, 0], self.xs[:, 1], z)

            ridge_splits = get_ridge_splits(vertices)
            test = [[] for _ in range(len(good_points))]
            for point, height in zip(self.xs, pred):
                region = check_region(point, ridge_splits)
                test[region].append(list(point) + [height])

            for ps in test:
                ps = np.array(ps)
                ax.scatter(ps[:, 0], ps[:, 1], ps[:,2])

            plt.show()
            print(vec)

        def vertices_from_state(state):
            assert len(state) % INPUT_DIM == 0
            num_vs = len(state) // INPUT_DIM
            vertices = state.reshape((num_vs, INPUT_DIM))
            return vertices

        def minimizer_error(state):
            vertices = vertices_from_state(state)
            try:
                return error(vertices)
            except self.BadSplitException:
                # I think this means the vertex locations were nonsensical
                # so just say the error here is bad
                return float('inf')

        simplex = get_unit_simplex(3)

        state_init = np.reshape(v.vertices, (np.product(v.vertices.shape)))
        simplex_init = state_init + get_unit_simplex(len(state_init)) * 1.0
        result = scipy.optimize.minimize(minimizer_error, np.zeros(state_init.shape),
                                         method='Nelder-Mead',
                                         options={'disp': True,
                                                  'initial_simplex': simplex_init})
        state_opt = result.x



        preds_opt = predict(vertices_from_state(state_opt))
        plot_attempt(preds_opt, vertices_from_state(state_opt))




        return

        regress()





        print()
        for k, x in v.__dict__.items():
            print(k)

        planes = []
        #for simplex_i, points_i in zip(v.ridge_vertices, v.ridge_points):
        #    tri = self.get_simplex(data, v.vertices, simplex_i, points_i)
        #    planes.append(tri)
        for simplex_i, points_i in zip(v.ridge_vertices, v.ridge_points):
            if -1 in simplex_i:
                continue
            planes.append([v.vertices[i] for i in simplex_i+[simplex_i[0]]])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:,0], data[:,1], data[:,2])
        ax.scatter(v.vertices[:,0], v.vertices[:,1], v.vertices[:,2])
        for tri in planes:
            #temp = np.array(tri + [tri[0]])
            temp = np.array(tri)
            ax.plot(temp[:,0], temp[:,1], temp[:,2])
        plt.show()

        exit()




    def get_simplex(self, data, vertices, simplex_i, points_i):
        assert len(points_i) == 2
        point_a = xs[points_i[0],:]
        point_b = xs[points_i[1],:]

        if -1 not in simplex_i:
            return vertices[simplex_i]

        # difference between points is normal
        # we already have 2 points in the simplex
        known_is = [i for i in simplex_i if i != -1]
        assert len(known_is) == 2
        v_a = vertices[known_is[0]]
        v_b = vertices[known_is[1]]
        new_dir = np.cross(point_a - point_b, v_a - v_b)
        new_point1 = (v_a + v_b) / 2 + new_dir*1.0
        new_point2 = (v_a + v_b) / 2 - new_dir*1.0
        return [v_a, new_point1, v_b, new_point2, v_a]










class CandidateFit:

    class Candidate:
        def __init__(self, cf, center, etol):
            self.cf = cf
            INPUT_DIM = self.cf.INPUT_DIM

            dists = np.linalg.norm(self.cf.xs - center, axis=1)
            ordering = np.argsort(dists)
            self.points_i = list(ordering[:INPUT_DIM + 1])
            data = self.cf.data[self.points_i]

            plane = None
            plane_point_count = 0
            total_count = 0
            regression_count = 0
            for point_i in ordering[INPUT_DIM + 1:]:
                new_row = self.cf.data[point_i]
                total_count += 1

                #if (plane_point_count > (INPUT_DIM+1)*2
                #    and data.shape[0] < 2*plane_point_count):
                #    rough_pred = new_row[:-1] @ plane
                #    rough_err_point = rough_pred - new_row[-1]
                #    #rough_error = rough_pred - new_row[-1]
                #    rough_error = (err**2 + rough_err_point**2)*TODO
                #    if abs(rough_error) < etol:
                #        # nice, just add with existing plane
                #        data = np.vstack((data, new_row))
                #    elif plane_point_count > abs(rough_error) > 2*etol:
                #        # don't bother with this one
                #        continue


                # if we made it here, we couldn't do the rough check
                regression_count += 1
                new_data = np.vstack((data, new_row))
                new_plane, residual, r, s = np.linalg.lstsq(
                    new_data[:, :-1], new_data[:, -1], rcond=None)
                preds = new_data[:, :-1] @ new_plane
                errs = preds - new_data[:, -1]
                #err = np.max(np.abs(errs))
                new_err = np.linalg.norm(errs)
                if new_err < etol:
                #if len(data) < 400:
                    data = new_data
                    self.points_i.append(point_i)
                    plane = new_plane
                    plane_point_count = new_data.shape[0]
                    err = new_err

            print(regression_count, '/', total_count)
            # mostly saved for plotting; maybe remove for mem performance?
            self.data = data

            plane, residual, r, s = np.linalg.lstsq(
                data[:, :-1], data[:, -1], rcond=None)
            self.plane = plane

            self.reset()

        def reset(self):
            self.points_remaining_i = self.points_i


        def plot(self):
            assert self.cf.INPUT_DIM == 2
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.cf.data[:, 0], self.cf.data[:, 1], self.cf.data[:, 3])

            #ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2]+3.0)
            plane, residual, r, s = np.linalg.lstsq(
                self.data[:, :-1], self.data[:, -1], rcond=None)
            preds = self.data[:, :-1] @ plane
            ax.scatter(self.data[:, 0], self.data[:, 1], preds)

            plt.show()
            print()


    def __init__(self, xs, z):
        #self.maxes = np.max(xs, axis=0)
        #self.mins = np.min(xs, axis=0)
        #self.xs = (xs - self.mins) / (self.maxes - self.mins)
        self.xs = xs
        self.z = z

        #self.assignment = np.full(self.xs.shape[0], -1)
        self.INPUT_DIM = self.xs.shape[1]
        self.NUM_POINTS = self.xs.shape[0]
        self.data = np.hstack((self.xs,
                               np.ones((self.NUM_POINTS, 1)),
                               self.z.reshape((self.NUM_POINTS, 1))))

        etol = 0.4
        NUM_ROUNDS = 5
        candidates = []
        scatter_etol = []
        scatter_points = []
        for i in range(NUM_ROUNDS):
            uncovered = np.array(range(self.NUM_POINTS))
            while len(uncovered) > 0:
                seed_i = np.random.choice(uncovered)

                #etol_choice = etol * 2**np.random.uniform(-3, 3)
                etol_choice = etol



                c = self.Candidate(self, self.xs[seed_i], etol_choice)
                candidates.append(c)
                scatter_etol.append(etol_choice)
                scatter_points.append(len(c.points_i))
                uncovered = np.setdiff1d(uncovered, c.points_i)

        #plt.figure()
        #plt.scatter(scatter_etol, scatter_points)
        #plt.show()

        #for i in range(NUM_CANDIDATES):
        #    x = np.random.rand(self.INPUT_DIM)
        #    c = self.Candidate(self, x, etol)
        #    candidates.append(c)


        points_remaining_i = np.array(range(self.NUM_POINTS))
        choices = []
        while True:
            choice_i = np.argmax([len(c.points_remaining_i) for c in candidates])
            print()
            print(sorted(-len(c.points_remaining_i) for c in candidates))
            print(len(points_remaining_i))
            print('choosing', choice_i)

            c = candidates.pop(choice_i)
            choices.append(c)

            print('Time:', time.time() - start_time)

            # Incremental plot of planes so far
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            #ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 3])
            for c_plot in choices:
                plane, residual, r, s = np.linalg.lstsq(
                    c_plot.data[:, :-1], c_plot.data[:, -1])
                preds = c_plot.data[:, :-1] @ plane
                ax.scatter(c_plot.data[:, 0], c_plot.data[:, 1], preds)
            plt.show()

            points_remaining_i = np.setdiff1d(points_remaining_i, c.points_remaining_i)
            if len(points_remaining_i) == 0:
                print('FIT USING ', len(choices), 'PLANES')

                break

            # update candidates for the next selection
            for candidate in candidates:
                candidate.points_remaining_i = np.setdiff1d(candidate.points_remaining_i, c.points_remaining_i)
        else:
            assert False, 'Oops, not enough candidates, didnt span everything'

        self.candidates = choices



class ContinuousFit:
    def __init__(self, xs, z, planes, subsets):
        NUM_POINTS = xs.shape[0]
        data = np.hstack((xs, z.reshape((NUM_POINTS, 1))))
        # use planes as a seed
        # assuming, for all i,j in bounds:
        # row === data[subsets[i][j], :]
        # plane === planes[i]
        # row[-1] ~= row[:-1] @ plane[:-1] + plane[-1]

        NUM_PLANES = len(planes)
        assert len(subsets) == NUM_PLANES
        INPUT_DIM = data.shape[1] - 1
        assert len(planes[0]) == INPUT_DIM + 1

        cuts = []
        for i, j in  itertools.combinations(range(NUM_PLANES), r=2):
            diff = planes[i] - planes[j]
            cuts.append(diff)

        def get_region(point):
            assert len(point) == INPUT_DIM
            sides = []
            for c in cuts:
                val = point @ c[:-1] + c[-1]
                sides.append(val > 0)
            return tuple(sides)

        # now for each region, count the number of subset points from each
        # subset. Basically, each region gets votes from various subsets (k)
        # for which subset best represents that region
        counts = defaultdict(lambda: defaultdict(lambda: 0))
        for k, subset in enumerate(subsets):
            for point_i in subset:
                point = data[point_i, :-1]
                region = get_region(point)
                counts[region][k] += 1
        region_to_subset = {}
        for region, votes in counts.items():
            winning_subset = max((count, subset)
                                 for subset, count in votes.items())[1]
            region_to_subset[region] = winning_subset

        #def predict(point):
        #    region = get_region(point)
        #    if region not in region_to_subset:
        #        return np.array([-1, -1])
        #    subset = region_to_subset[region]
        #    # subset is also plane_i
        #    plane = planes[subset]
        #    pred = point @ plane[:-1] + plane[-1]
        #    return np.array([subset, pred])
        #predict_vec = np.vectorize(predict, signature='(n)->(2)')

        def predict(point):
            region = get_region(point)
            if region not in region_to_subset:
                return 1, -1
            subset = region_to_subset[region]
            # subset is also plane_i
            plane = planes[subset]
            pred = point @ plane[:-1] + plane[-1]
            return subset, pred
        predict_vec = np.vectorize(predict, signature='(n)->(),()', otypes=[int, np.float64])

        # stress test, I guess
        print('beginning stress test')
        NUM_STRESS = 1000
        new_x = np.random.rand(NUM_STRESS) * 5
        new_y = np.random.rand(NUM_STRESS) * 5
        new_input = np.stack((new_x, new_y), 1)
        new_subsets, new_preds = predict_vec(new_input)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(new_input[:,0], new_input[:,1], new_preds, c=new_subsets)
        plt.show()

        print()




class SimplexFit:
    '''
    If we want T triangles, how many vertices do we need?
    F+V=E+2
    T = F-1
    E = (T*3+X)/2, X is number of edges on convex hull
    T+1+V-2 = T*3/2+X/2
    1+V-2-X/2=T/2
    T = 2V - X - 2

    '''

    def __init__(self, xs, z, boundaries):
        INPUT_DIM = xs.shape[1]
        self.INPUT_DIM = INPUT_DIM
        NUM_POINTS = xs.shape[0]
        NUM_VS = NUM_POINTS // 4
        #NUM_VS = 16
        boundaries = np.array(boundaries)
        self.xs = xs
        self.z = z

        EDGE_NUM = int(NUM_VS**(1/INPUT_DIM))
        #EDGE_NUM = 2
        '''
        For INPUT_DIM=N, There are 2*N boundaries, each an N-1 dimensional square
        We break each boundary into EDGE_NUMxEDGE_NUMx... regions
        Each boundary uses (EDGE_NUM-1)^ uhhh
        Actually, ignore that. If we were to fill the hypercube with
        evenly-spaced points, we would need (EDGE_NUM+1)^N, but we will remove
        the inside (EDGE_NUM-1)^N. This leaves roughly 2*N*(EDGE_NUM)^(N-1)
        '''
        # this loop is order (EDGE_NUM-1)^INPUT_DIM, could be slightly better
        edges = []
        for i, vertex in enumerate(itertools.product(np.linspace(0,1,EDGE_NUM),repeat=INPUT_DIM)):
            if (0 not in vertex) and (1 not in vertex):
                continue
            edges.append(vertex)
        edges = np.array(edges)

        #other = Sampler.get_orthogonal_samples(INPUT_DIM, NUM_VS-edges.shape[0])
        other = np.random.random((NUM_VS-edges.shape[0], INPUT_DIM))
        #other = np.array([[0, 0.16001],
        #                  [0.26663, 0],
        #                  [0.80002, 1],
        #                  [1, 0.79999]])

        vertices_unscaled = np.vstack((edges, other))

        #vertices_unscaled = np.array([
        #    [.2, -3],
        #    [.2, 3],
        #    [-20, .5],
        #    [.8, -3],
        #    [.8, 3],
        #    [20, 0.5],
        #    [0.5, 0.5]
        #])

        vertices = vertices_unscaled*(boundaries[1]-boundaries[0])+boundaries[0]

        print('Initial test')
        self.evaluate_vertices(vertices, plot=True)

        old_num = float('inf')
        DECAY_RATE = 0.75
        END_NUM = 8
        assert END_NUM >= 2**INPUT_DIM
        MINIMIZER_START = 12 # good with 30
        vertices_history = []
        error_history = []
        while True:
            d, heights, preds = self.evaluate_vertices(vertices)
            print(f'\nTop of loop with {len(vertices)} vertices, {d.simplices.shape[0]} simplices')
            err = np.linalg.norm(preds - self.z)
            vertices_history.append(vertices.shape[0])
            error_history.append(err)

            if len(vertices) <= 0:
                #d, heights, preds = self.evaluate_vertices(vertices)
                #fig = plt.figure()
                #ax = fig.add_subplot(projection='3d')
                #ax.scatter(vertices[:, 0], vertices[:, 1],
                #           np.clip(heights, -10, 10))
                #color = d.find_simplex(xs)
                ## ax.scatter(xs[:,0], xs[:,1], preds, c=color)

                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(xs[:, 0], xs[:, 1], z)
                ax.scatter(vertices[:, 0], vertices[:, 1], heights)
                color = d.find_simplex(xs)
                ax.scatter(xs[:, 0], xs[:, 1], preds, c=color)
                plt.show()


            old_num = len(vertices)
            if old_num == END_NUM:
                break
            new_num = round(old_num*DECAY_RATE)
            if new_num < END_NUM:
                new_num = END_NUM

            vertices = self.cull_vertices(vertices, new_num, d, heights)
            assert vertices.shape[0] == new_num
            print('After culling to', new_num)
            d, heights, preds = self.evaluate_vertices(vertices, plot=True)


            if new_num <= MINIMIZER_START:
                DECAY_RATE = 0.866
                #print('calling minimizer, with', new_num, 'vertices')
                vertices = self.wiggle(vertices, z, final=new_num==END_NUM)

                d, heights, preds = self.evaluate_vertices(vertices, plot=True)



        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(vertices[:, 0], vertices[:, 1],
        #           np.clip(heights, -10, 10))
        #color = d.find_simplex(xs)
        ## ax.scatter(xs[:,0], xs[:,1], preds, c=color)

        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(xs[:, 0], xs[:, 1], z)
        #color = d.find_simplex(xs)
        #ax.scatter(xs[:, 0], xs[:, 1], preds, c=color)
        #ax.scatter(vertices[:, 0], vertices[:, 1], heights)
        ##plt.show()

        ## here are the figures I usually use
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #color = d.find_simplex(xs)
        #ax.scatter(xs[:, 0], xs[:, 1], preds - z, c=color)
        ##plt.show()

        #fig = plt.figure()
        #plt.grid()
        #plt.loglog(vertices_history, error_history, 'x')
        ##plt.show()


        #plt.show()


        fit = Fit.from_triangulation(d, heights)
        self.fit = fit
        print('done')

        ## delete the ones with low angles
        #lowest_angles = np.argsort(ratings)[:round(NUM_VS*0.9)]
        #for c in corners_i:
        #    if c in lowest_angles:
        #        lowest_angles = np.delete(lowest_angles, np.where(lowest_angles==c))
        #culled_vertices = np.delete(vertices, lowest_angles, axis=0)

        #d, heights, preds = self.evaluate_vertices(culled_vertices)

        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(xs[:,0], xs[:,1], z)
        #color = d.find_simplex(xs)
        #ax.scatter(xs[:,0], xs[:,1], preds, c=color)
        #plt.show()


    def get_position(self, d, point):
        simplex_i = int(d.find_simplex(point))
        if simplex_i == -1:
            # this sample point is outside the space spanned by the vertices
            # there's an issue with the vertices; probably missing corners
            #assert False
            return None, None

        vertex_is = d.simplices[simplex_i]
        # barycentric = d.transform[simplex_i] @ point
        point_shifted = point - d.transform[simplex_i][-1]
        barycentric_short = d.transform[simplex_i][:-1] @ point_shifted
        barycentric = np.append(barycentric_short, 1 - sum(barycentric_short))

        assert sum(barycentric) == 1
        for b in barycentric:
            # TODO when there are quantized input dimensions you often have
            # points exactly on the edge, and they introduce rounding errors
            assert -1e-12 <= b <= 1+1e-12

        return vertex_is, barycentric

    def evaluate_vertices(self, vertices, plot=False):
        NUM_VS = vertices.shape[0]

        d = scipy.spatial.Delaunay(vertices)

        if (plot and NUM_VS <= 8
            and d.points.shape[1] == 2):
            figure = plt.figure()
            ax = figure.add_subplot()
            #ax.plot([0, 5], [5, 15/8], '--')
            scipy.spatial.delaunay_plot_2d(d, ax)
            #plt.show()

        # now we build matrix M s.t.
        # M @ heights = pred
        # and we can say
        # pinv(M) @ z = heights_best_fit
        # this relies on knowing which triangle each row of z is in
        M = []
        for point in self.xs:
            vertex_is, barycentric = self.get_position(d, point)
            if barycentric is None:
                # NOTE this is intended behavior when rate_vertices2 tries to
                # remove a corner
                return d, None, np.array([float('inf')] * self.z.shape[0])

            row = [0]*NUM_VS
            for vertex_i, b in zip(vertex_is, barycentric):
                row[vertex_i] = b
            M.append(row)
        M = np.array(M)

        heights = np.linalg.pinv(M) @ self.z
        preds = M @ heights


        err = np.linalg.norm(preds - self.z)
        if plot:
            print('got err', err)

        return d, heights, preds

    def rate_vertices(self, d, heights):
        # lower is better for removal
        INPUT_DIM = self.INPUT_DIM
        NUM_VS = len(d.points)
        assert len(heights) == NUM_VS
        # quick note on notation: var_i is an index into d.simplices or
        # d.points. var_index is an index into d.simplices[foo_i], so it's the
        # second indirection
        vertices = np.hstack((d.points, heights.reshape(NUM_VS, 1)))

        badness_by_vertex = np.zeros(NUM_VS)
        for simplex_a_i, a_neighbors in enumerate(d.neighbors):
            for a_unshared_index, simplex_b_i in enumerate(a_neighbors):
                if simplex_b_i == 0:
                    # we are at the edge
                    continue
                if simplex_a_i > simplex_b_i:
                    # avoid checking this combo twice
                    continue
                a_unshared_i = d.simplices[simplex_a_i][a_unshared_index]
                # Let's find the angle between these a and b simplices
                b_neighbors = d.neighbors[simplex_b_i]
                b_unshared_index = list(b_neighbors).index(simplex_a_i)
                b_unshared_i = d.simplices[simplex_b_i][b_unshared_index]

                all_vertices_a_i = d.simplices[simplex_a_i]
                shared_vertices_i = np.delete(all_vertices_a_i,
                                              a_unshared_index)

                for test_shared in d.simplices[simplex_b_i]:
                    assert (test_shared == b_unshared_i
                            or test_shared in shared_vertices_i)

                shared_vertices = vertices[shared_vertices_i]

                ref_point = shared_vertices[0]
                # subspace uses column vectors
                shared_subspace = (shared_vertices[1:] - ref_point).T
                a_vector = vertices[a_unshared_i] - ref_point
                b_vector = vertices[b_unshared_i] - ref_point
                assert shared_subspace.shape == (INPUT_DIM+1, INPUT_DIM-1)

                unshared_subspace = np.linalg.qr(shared_subspace, mode='complete')[0][:, -2:]
                # this matrix multiply both projects the vectors onto the
                # unshared subspace and converts them to a (2D) coordinate
                # system that only spans the unshared subspace
                a_subspace = unshared_subspace.T @ a_vector
                b_subspace = unshared_subspace.T @ b_vector

                # now we get the angle normally (clip b/c of precision issue)
                norm = np.linalg.norm
                a_norm = norm(a_subspace)
                b_norm = norm(b_subspace)
                if a_norm == 0 or b_norm == 0:
                    # one of the simplices has zero volume ... that's weird
                    # I saw it when some subspaces had no datapoints
                    # in that case we don't need vertices here; say it's the
                    # flattest possible angle, so vectors are antiparallel
                    x = -1.0
                else:
                    x = (np.dot(a_subspace, b_subspace)
                         / (a_norm*b_norm))
                x_safe = np.clip(x, -1.0, 1.0)
                angle = np.arccos(x_safe)

                ## TEST
                #assert INPUT_DIM == 2
                #normal_a = np.cross(shared_subspace.reshape(3), a_vector)
                #normal_b = np.cross(shared_subspace.reshape(3), b_vector)
                #y = (np.dot(normal_a, normal_b)
                #     / (norm(normal_a) * norm(normal_b)))
                #y_safe = np.clip(y, -1.0, 1.0)
                #test_angle = np.arccos(y_safe)
                #assert abs(angle - test_angle) < 1e-5

                # pi means simplices agree, pi/2 means 90 degrees off,
                # 0 means an incredibly sharp fold
                badness = np.pi - angle
                assert 0 <= badness <= np.pi
                badness_by_vertex[shared_vertices_i] += badness


        return badness_by_vertex


    def rate_vertices2(self, d):
        # dead simple: try removing each vertex, and see what happens
        orig_vertices = d.points
        ans = []
        for i in range(len(orig_vertices)):
            new_vertices = np.delete(orig_vertices, i, 0)
            preds = self.evaluate_vertices(new_vertices)[2]
            err = np.linalg.norm(preds - self.z)
            ans.append(err)
        return np.array(ans)


    def cull_vertices(self, vertices, desired_num, d, heights):
        if vertices.shape[0] < 100:
            ratings = self.rate_vertices2(d)
        else:
            ratings = self.rate_vertices(d, heights)

        bounds = 0, 1
        corners_i = [i for i, v in enumerate(vertices)
                     if all(x == bounds[0] or x == bounds[1] for x in v)]
        #print(f'Found {len(corners_i)} corners')

        # delete the ones with low angles
        ratings_i = np.argsort(ratings)
        for c in corners_i:
            if c in ratings_i:
                ratings_i = np.delete(ratings_i, np.where(ratings_i==c))
        lowest_angles = ratings_i[:len(vertices)-desired_num]

        culled_vertices = np.delete(vertices, lowest_angles, axis=0)


        test_corners_i = [i for i, v in enumerate(vertices)
                     if all(x == bounds[0] or x == bounds[1] for x in v)]
        assert len(test_corners_i) == len(corners_i)

        return culled_vertices

    def wiggle(self, vertices, z, final=False):
        INPUT_DIM = vertices.shape[1]
        # use nelder-mead to adjust positions
        # You can move things along a boundary, but not off a boundary
        mins = np.min(vertices, 0)
        maxs = np.max(vertices, 0)
        # TODO I am relying on floating point comparison here, but I think
        # it should be okay since the values were arrived at the same way
        bottoms = vertices == 0
        tops = vertices == 1
        not_boundaries = np.logical_not(np.logical_or(bottoms, tops))
        if np.count_nonzero(not_boundaries) == 0:
            # everything is jammed in a corner?
            return vertices
        vertices_orig = vertices.copy()

        def vs_to_state(vertices):
            # don't move things off an edge:
            state = np.extract(not_boundaries, vertices)
            return state

        def state_to_vs(state):
            vs = vertices_orig.copy()
            np.place(vs, not_boundaries, state)
            return vs

        def minimizer_error(state):
            # TODO penalize vs outside the boundaries
            vs = state_to_vs(state)
            d, heights, preds = self.evaluate_vertices(vs)
            err = np.linalg.norm(preds-z)
            return err

        state_init = vs_to_state(vertices_orig)
        print('Going into minimizer, error', minimizer_error(state_init))
        converted_vs = state_to_vs(state_init)
        assert np.all(converted_vs == vertices_orig)

        state_init = vs_to_state(vertices_orig)
        simplex = get_unit_simplex(len(state_init))
        INIT_SIMPLEX_SIZE = 0.3
        #simplex_stretch = simplex * INIT_SIMPLEX_SIZE*(maxs-mins) + mins
        # TODO to know how much to stretch the simplex in each dimension you
        # need to know which input dimension each state entry came from
        # TODO while you're at it, also pass those bounds in the options
        simplex_stretch = simplex * INIT_SIMPLEX_SIZE
        simplex_init = state_init + simplex_stretch
        max_fev = 1000 if final else 100
        result = scipy.optimize.minimize(minimizer_error, np.zeros(state_init.shape),
                                         bounds=[[0,1]]*len(state_init),
                                         method='Nelder-Mead',
                                         options={'disp': True,
                                                  'maxfev': max_fev,
                                                  'initial_simplex': simplex_init,
                                                  #'fatol': 0.1,
                                                  'adaptive': True})

        result_state = result.x
        #print('quick test', minimizer_error(result_state))
        result_vertices = state_to_vs(result_state)

        # sometimes two vertices get pushed into a corner together - in that
        # case they are redundant
        result_vertices_clean = np.unique(result_vertices, axis=0)
        return result_vertices_clean

    def predict_one(self, d, heights, point):
        vertex_is, barycentric = self.get_position(d, point)
        vertex_heights = point[vertex_is]
        ans = vertex_heights @ barycentric
        return ans

class Fit:
    def __init__(self, edges, planes, mapping):
        # mostly we use homogeneous coordinates for the point
        # "edges" is a matrix where each row represents one region division
        #     an "edge" is an N-1 dim dividing plane in N dim input space
        # "planes" is a matrix where each row is a fitting plane
        #     a "plane" is an N dim fitting plane in N+1 dim input/output space
        # "mapping" is a list where each entry defines the requirements to be
        # considered within a particular region.
        # For more documentation, look at Fit.predict
        self.edges = edges
        self.planes = planes
        self.mapping = mapping

        self.INPUT_DIM = self.edges.shape[1] - 1
        assert self.planes.shape[1] == self.INPUT_DIM + 1


    def get_region(self, point_h):
        edge_decisions = (self.edges @ point_h) > 0

        matching_regions = []
        for plane_i, map_pairs in enumerate(self.mapping):
            if all(edge_decisions[edge_i] == is_positive
                   for edge_i, is_positive in map_pairs):
                matching_regions.append(plane_i)

        # if the mapping was constructed correctly, then a point inside will
        # always match exactly one region. There may be issues if the point is
        # exactly on a vertex; I haven't thought that fully through. Should be
        # very careful with that in the limited-bitwidth verilog implementation
        assert len(
            matching_regions) > 0, 'No matching regions, point may be outside bounds?'
        #assert len(
        #    matching_regions) < 2, 'Multiple matching regions, probably bug in mapping'
        if len(matching_regions) > 1:
            print('MULTIPLE MATCHING REGIONS FOR', point_h)
        matching_region = matching_regions[0]
        return matching_region

    def predict(self, point):
        assert point.shape == (self.INPUT_DIM,)
        # using homogeneous coordinates
        point_h = np.append(point, 1)
        matching_region = self.get_region(point_h)
        plane = self.planes[matching_region]
        height = plane @ point_h
        return height

    def predict_vec(self, xs):
        pv = np.vectorize(lambda point: self.predict(point),
                          signature='(n)->()')
        return pv(xs)


    @classmethod
    def from_triangulation(cls, d, heights):
        INPUT_DIM = d.points.shape[1]
        class Region:
            def __init__(self, plane, vertices_i):
                self.edges = []
                self.planes = [plane]
                self.vertices_i = set(vertices_i)

            def get_plane(self):
                # CAREFUL! This is only a good average near (0,0), but because
                # our epsilon is so small it should be fine throughout our
                # 0-1 square as well
                avg = np.average(np.array(self.planes), axis=0)
                return avg

            #def absorb(self, other):
            #    self.planes += other.planes
            #    self.vertices_i |= other.vertices_i
            #    for e in other.edges:
            #        if e.region_a is other:
            #            e.region_a = self
            #        if e.region_b is other:
            #            e.region_b = self

            def __str__(self):
                return str(id(self))[-4:]

        class Edge:
            def __init__(self, region_a, region_b, positive):
                # region order doesn't matter
                self.region_a = region_a
                self.region_b = region_b
                # positive iff we take the max of the regions at the break
                self.positive = positive

                self.region_a.edges.append(self)
                self.region_b.edges.append(self)

            def remove(self):
                # region a absorbs region b
                # edges referencing b should reference a instead
                a = self.region_a
                b = self.region_b
                a.planes += b.planes
                a.vertices_i |= b.vertices_i

                a.edges.remove(self)
                for e in b.edges:
                    if e is self:
                        continue
                    if e.region_a == b:
                        e.region_a = a
                    if e.region_b == b:
                        e.region_b = a
                    a.edges.append(e)



            def __str__(self):
                return f'({self.region_a}, {self.region_b})'

        # use d.simplices and heights to initialize all the regions with their
        # fitting planes
        regions = []
        for vertex_list_i in d.simplices:
            vertices_input = d.points[vertex_list_i]
            vertices_output = heights[vertex_list_i].reshape((len(vertex_list_i),1))
            vertices = np.concatenate((vertices_input, vertices_output), axis=1)
            vectors = vertices[1:] - vertices[0]
            normal = np.linalg.qr(vectors.T, mode='complete')[0][:,-1]
            dot_result = normal @ vertices[0]
            # we now have normal@xyz = dotresult
            # ax+by+cz=d
            # z=(-ax-by+d)/c
            plane = np.append(-normal[:-1], dot_result) / normal[-1]
            #plane = np.append(normal, -dot_result)
            region = Region(plane, vertex_list_i)
            regions.append(region)

        # use d.neighbors to populate regions/edges graph
        edges = []
        for region_i, neighbor_list in enumerate(d.neighbors):
            for opposite_vertex_index, neighbor_i in enumerate(neighbor_list):
                if neighbor_i < region_i:
                    # avoid doing this pair twice
                    continue
                opposite_vertex_i = d.simplices[region_i][opposite_vertex_index]

                # test_point is inside (edge of) region and not inside neighbor
                test_point = np.append(d.points[opposite_vertex_i], 1)
                region_height = regions[region_i].planes[0] @ test_point
                neighbor_height = regions[neighbor_i].planes[0] @ test_point
                positive_edge = region_height > neighbor_height

                e = Edge(regions[region_i], regions[neighbor_i], positive_edge)
                edges.append(e)


        # go through all the edges and delete the unnecesary ones
        EPSILON = 1e-3
        necessary_edges = []
        for e in edges:
            if e.region_a is e.region_b:
                # happens if we delete all edges around a vertex
                continue
            diff = e.region_a.get_plane() - e.region_b.get_plane()
            fit_check = np.linalg.norm(diff) < EPSILON
            combined_vertices_i = e.region_a.vertices_i | e.region_b.vertices_i
            combined_vertices = d.points[list(combined_vertices_i)]
            qhull_options = f'{"Qx" if INPUT_DIM > 4 else ""} Qc'
            hull = scipy.spatial.ConvexHull(combined_vertices,
                                                   qhull_options=qhull_options)
            # TODO If we ignore "coplanar" then we are just checking that it's
            # convex, when we should be checking if it's nonconcave. I added
            # the option "Qc" and checked hull.coplanar to get around that, but
            # I'm not 100% sure I did it right
            convex_check = (len(hull.vertices) + len(hull.coplanar)
                            == len(combined_vertices_i))
            if fit_check and convex_check:
                # combine
                e.remove()
            else:
                # keep
                necessary_edges.append(e)


        necessary_regions = set()
        for e in necessary_edges:
            necessary_regions.add(e.region_a)
            necessary_regions.add(e.region_b)
        necessary_regions = list(necessary_regions)

        final_edges = []
        for e in necessary_edges:
            # ax+by+c = dx+ey+f
            # (a-d)x + (b-e)y + (c-f) = 0
            eq = e.region_a.get_plane() - e.region_b.get_plane()
            # a positive dot product with eq means a > b
            final_edges.append(eq)
        final_edges = np.array(final_edges)

        final_regions = []
        for r in necessary_regions:
            final_regions.append(r.get_plane())
        final_regions = np.array(final_regions)

        final_mapping_dict = defaultdict(list)
        for e_i, e in enumerate(necessary_edges):
            a = e.region_a
            a_i = necessary_regions.index(a)
            final_mapping_dict[a].append((e_i, e.positive))
            b = e.region_b
            b_i = necessary_regions.index(b)
            final_mapping_dict[b].append((e_i, not e.positive))
        final_mapping = [final_mapping_dict[r] for r in necessary_regions]

        return cls(final_edges, final_regions, final_mapping)







def presentation_plot():
    x, y, z = get_data2()
    y = 5-y
    plot(x, y, z)

    x, y, z = get_data5()
    x = x/5
    y = y/5
    plot(x, y, z)

    z = z + 2*x
    plot(x, y, z)

    plt.show()



start_time = time.time()
if __name__ == '__main__':
    #presentation_plot()
    #exit()
    #fit(*get_data())
    #exit()

    #x, y, z = get_data()
    #fit3(np.stack((x.T, y.T), 1), z)

    #fit4(*get_data())

    x, y, z = get_data4()
    xs = np.stack((x.T, y.T), 1)

    #FragmentFit(xs, z)
    #NetFit(xs, z)
    #cf = CandidateFit(xs, z)
    #planes = [c.plane for c in cf.candidates]
    #subsets = [c.points_i for c in cf.candidates]

    #ContinuousFit(xs, z, planes, subsets)


    xy = xs[:,0]*xs[:,1]
    #xy = xy * 5 / np.max(xy)
    xs = np.insert(xs, 2, xy, axis=1)
    xs = xs[:,1:3]
    fit = SimplexFit(xs, z, [[0,0], [1,1]]).fit
    #fit = SimplexFit(xs, z, [[0,0,0], [5,5,5]]).fit
    preds = fit.predict_vec(xs)
    fig = plt.figure()

    #plt.scatter(xs[:,2], z)
    #plt.plot(xs[:,2], preds)
    #plt.show()

    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs[:,0], xs[:,1], z)
    #color = d.find_simplex(xs)
    color = [fit.get_region(point_h)
             for point_h in np.append(xs, np.ones((len(xs), 1)), 1)]
    ax.scatter(xs[:,0], xs[:,1], preds, c=color)


    def get_xs(gain):
        N = 100
        gain = np.ones((N, 1)) * gain
        input = np.linspace(0, 1, N).reshape((N, 1))
        # return np.hstack((input, gain))
        return np.hstack((input, gain, input * gain / 1))
        # return np.hstack((gain, input * gain / 5))


    plt.figure()
    for gain in [0, .25, .5, .75, 1]:
        xs_full = get_xs(gain)
        preds = data_transform_4(xs_full[:, 0], xs_full[:, 1])
        plt.plot(xs_full[:, 0], preds)
    plt.grid()
    # plt.show()

    plt.figure()
    for gain in [0, .25, .5, .75, 1]:
        xs_full = get_xs(gain)
        preds = fit.predict_vec(xs_full[:, 1:3])
        plt.plot(xs_full[:, 0], preds)
    plt.grid()

    plt.show()
    print()
    #wrapper = lambda _: SimplexFit(xs, z, [[0,0], [1,1]])
    #cProfile.run('wrapper(0)')

    #SimplexFit(xs, z, [[0,0], [1,1]])

    #xs, z = get_data_3d()
    #xs, z = get_data_binary()
    #SimplexFit(xs, z, [[0,0], [1,1]])
    #SimplexFit(xs, z, [[0,0,0,0], [1,1,1,1]])

    # check amp with generated dimension
    #SimplexFit(xs, z, [[0,0], [1,1]])
    # without ext, some data:
    # 10 vertices -> 1.59
    # 7 vertices -> 2.12
    # 5 vertices -> 4.51

    xy = xs[:,0]*xs[:,1]
    #xy = xy / np.max(xy)
    xs_ext = np.insert(xs, 2, xy, axis=1)
    SimplexFit(xs_ext, z, [[0,0,0], [1,1,1]])
    print()