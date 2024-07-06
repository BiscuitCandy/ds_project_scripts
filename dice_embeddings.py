import numpy as np
import math
import csv

class DICE:
    '''
    DICE class turns numbers into their respective DICE embeddings
    
    Since the cosine function decreases monotonically between 0 and pi, simply employ a linear mapping
    to map distances s_n \in [0, |a-b|] to angles \theta \in [0, pi]
    '''
    def init(self, d=2, min_bound=0, max_bound=100, norm="l2"):
        self.d = d # By default, we build DICE-2
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.norm = norm  # Restrict x and y to be of unit length
        self.M = np.random.normal(0, 1, (self.d, self.d))
        self.Q, self.R = np.linalg.qr(self.M, mode="complete")  # QR decomposition for orthonormal basis, Q
    
    def linear_mapping(self, num):
        '''Eq. (4) from DICE'''
        norm_diff = num / abs(self.min_bound - self.max_bound)
        theta = norm_diff * math.pi
        return theta
    
    def make_dice(self, num):
        r = 1
        theta = self.linear_mapping(num)
        if self.d == 2:
            # DICE-2
            polar_coord = np.array([r*math.cos(theta), r*math.sin(theta)])
        elif self.d > 2:
            # DICE-D
            polar_coord = np.array([(math.sin(theta)**(dim-1)) * math.cos(theta) if dim < self.d else (math.sin(theta)**(self.d)) for dim in range(1, self.d+1)])
        else:
            raise ValueError("Wrong value for d. d should be greater than or equal to 2.")
            
        dice = np.dot(self.Q, polar_coord)  # DICE-D embedding for num
        
        dice =  dice.tolist()
        return dice

dice = DICE()
dice.init(d=384, min_bound=0, max_bound=100)

dice_file = open("./embeddings_numerics/cancerkg2_meta_v_dice.tsv", 'w+')

dice_writer = csv.writer(dice_file, delimiter="\t")

with open("meta_v_output", 'r+') as f :
    
    for i in f.readlines() :
        range_ = i.strip()
        l, r = map(float, range_[1:-1].split("|"))
        x, y = dice.make_dice(l), dice.make_dice(r)
        
        dice_writer.writerow([*x, *y])