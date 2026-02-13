import Einlean

namespace Einlean

-- Forward-only einops-style pipeline demo.
-- Backward/autograd is not implemented in Einlean yet.

def fb := dim! 2
def fc := dim! 3
def fh := dim! 2
def fw := dim! 2

def x : Tensor [fb, fc, fh, fw] Int := arange 1

def y0 : Tensor [fb, fc, fh, fw] Int := x

def y1 : Tensor [fb, fc] Int := y0.reduce .max

def y2 : Tensor [fc, fb] Int := y1.rearrange

def y3 : Tensor [] Int := y2.reduce .sum

#eval y0.shape
#eval y1.shape
#eval y2.shape
#eval y3.shape

#eval y1
#eval y2
#eval y3

end Einlean
