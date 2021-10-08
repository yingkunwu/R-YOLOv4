import torch
from shapely.geometry import Polygon


def R(theta):
    """
    Args:
        theta: must be radian
    Returns: rotation matrix
    """
    r = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]])
    return r


def T(x, y):
    """
    Args:
        x, y: values to translate
    Returns: translation matrix
    """
    t = torch.tensor([[1, 0, x], [0, 1, y], [0, 0, 1]])
    return t


def rotate(center_x, center_y, a, p):
    P = torch.matmul(T(center_x, center_y), torch.matmul(R(a), torch.matmul(T(-center_x, -center_y), p)))
    return P[:2]


def xywha2xyxyxyxy(p):
    """
    Args:
        p: 1-d tensor which contains (x, y, w, h, a)
    Returns: bbox coordinates (x1, y1, x2, y2, x3, y3, x4, y4) which is transferred from (x, y, w, h, a)
    """
    x, y, w, h, a = p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]

    x1, y1, x2, y2 = x + w / 2, y - h / 2, x + w / 2, y + h / 2
    x3, y3, x4, y4 = x - w / 2, y + h / 2, x - w / 2, y - h / 2

    P1 = torch.tensor((x1, y1, 1)).reshape(3, -1)
    P2 = torch.tensor((x2, y2, 1)).reshape(3, -1)
    P3 = torch.tensor((x3, y3, 1)).reshape(3, -1)
    P4 = torch.tensor((x4, y4, 1)).reshape(3, -1)
    P = torch.stack((P1, P2, P3, P4)).squeeze(2).T
    P = rotate(x, y, a, P)
    X1, X2, X3, X4 = P[0]
    Y1, Y2, Y3, Y4 = P[1]

    return X1, Y1, X2, Y2, X3, Y3, X4, Y4


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def skewiou(box1, box2):
    assert len(box1) == 5 and len(box2[0]) == 5
    iou = []
    g = torch.stack(xywha2xyxyxyxy(box1))
    g = Polygon(g.reshape((4, 2)))
    for i in range(len(box2)):
        p = torch.stack(xywha2xyxyxyxy(box2[i]))
        p = Polygon(p.reshape((4, 2)))
        if not g.is_valid or not p.is_valid:
            print("something went wrong in skew iou")
            return 0
        inter = Polygon(g).intersection(Polygon(p)).area
        union = g.area + p.area - inter
        iou.append(torch.tensor(inter / (union + 1e-16)))
    return torch.stack(iou)


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names
