import sys, math
import numpy as np
import numpy.linalg as la

# default orientation spring constant value (hartree/bohr)
#k_ref = 0000.0 / p4const.psi_hartree2kcalmol

# magnitude of numerical displacement during a finite difference derivative
fin_diff = 1.0 * 10**(-3)

# threshold for degenerate principal moments of inertia (1 part per 10**n)
thresh = 2

# absolute threshold for moment of inertia difference
mom_min = 2.0 * 10**(-2)

# absolute threshold for coordinate comparison difference (bohr)
cmin = 3.0 * 10**(-2)

# cartesian indices
xyz = {0: 'X', 1: 'Y', 2: 'Z', 'X': 0, 'Y': 1, 'Z': 2}

# global origin and global axes
global_origin = np.zeros(3)
global_axes =  np.identity(3)

## MATH FUNCTIONS ##

# compare if two values are the same within specified threshold
def are_same(n1, n2, tol, minval):
  same = False
  nmax = max(abs(n1), abs(n2), abs(minval))
  comp = abs((n2 - n1) / nmax)
  if (comp <= 10**(-tol)):
    same = True
  return same

# calculate distance between two 3-d cartesian coordinates
def get_r12(coords1, coords2):
  r2 = 0.0
  for p in range(3):
    r2 += (coords2[p] - coords1[p])**2
  r = math.sqrt(r2)
  return r

# calculate unit vector between to 3-d cartesian coordinates
def get_u12(coords1, coords2):
  r12 = get_r12(coords1, coords2)
  u12 = np.zeros(3)
  if (r12 > 0.0):
    for p in range(3):
      u12[p] = (coords2[p] - coords1[p]) / r12
  return u12

# calculate dot product between two unit vectors
def get_udp(uvec1, uvec2):
  udp = 0.0
  for p in range(3):
    udp += uvec1[p] * uvec2[p]
  udp = max(min(udp, 1.0), -1.0)
  return udp

# calculate unit cross product between two unit vectors
def get_ucp(uvec1, uvec2):
  ucp = np.zeros(3)
  cos_12 = get_udp(uvec1, uvec2)
  sin_12 = math.sqrt(1 - cos_12**2)
  if (sin_12 > 0.0):
    ucp[0] = (uvec1[1]*uvec2[2] - uvec1[2]*uvec2[1]) / sin_12
    ucp[1] = (uvec1[2]*uvec2[0] - uvec1[0]*uvec2[2]) / sin_12
    ucp[2] = (uvec1[0]*uvec2[1] - uvec1[1]*uvec2[0]) / sin_12
  return ucp

## CLASSES ##

# atom class for atomic data
class atom:
  # constructor
  def __init__(self, at_type, coords, charge, mass):
    self.attype = at_type
    self.coords = coords
    self.charge = charge
    self.mass = mass
    self.dists = {}

# molecule class for molecular data
class molecule:
  # constructor
  def __init__(self, psi4_molecule, k_ref):
    self.get_geom(psi4_molecule)
    self.k_ref = k_ref

  # read in geometry from psi4 input
  def get_geom(self, psi4_mol):
    #coords = psi4_mol.geometry()
    self.n_atoms = psi4_mol.natom()
    self.atoms = []
    for i in range(self.n_atoms):
      coords = [psi4_mol.x(i), psi4_mol.y(i), psi4_mol.z(i)]
      at_type = psi4_mol.symbol(i)
      charge = psi4_mol.charge(i)
      mass = psi4_mol.mass(i)
      self.atoms.append(atom(at_type, coords, charge, mass))
      self.atoms[i].rank = [i for j in range(6)]

  # update geometry from psi4 molecule
  def update_geom(self, psi4_mol):
    for i in range(self.n_atoms):
        self.atoms[i].coords = [psi4_mol.x(i), psi4_mol.y(i), psi4_mol.z(i)]

  # calculate center of mass of molecule
  def get_com(self):
    self.com = np.zeros(3)
    self.mass = 0.0
    for i in range(self.n_atoms):
      self.mass += self.atoms[i].mass
      for j in range(3):
        self.com[j] += self.atoms[i].mass * self.atoms[i].coords[j]
    if (self.mass > 0.0):
      for j in range(3):
        self.com[j] /= self.mass
    else:
      self.com = np.zeros(3)

  # calculate moment of inertia tensor of molecule
  def get_moi(self, point, axes):
    self.get_local_coords(point, axes)
    self.moi = np.zeros((3, 3))
    for i in range(self.n_atoms):
      atmass = self.atoms[i].mass
      for p in range(3):
        for q in range(3):
          if (p == q):
            r = (p+1) % 3
            s = (p+2) % 3
            val1 = self.local_coords[i][r]
            val2 = self.local_coords[i][s]
            self.moi[p][p] += atmass * (val1**2 + val2**2)
          else:
            val1 = self.local_coords[i][p]
            val2 = self.local_coords[i][q]
            self.moi[p][q] += -atmass * val1 * val2

  # determine inertial axes which diagonalize the moment of inertia tensor
  def get_inertial_axes(self):
    self.pmoms, self.axes = la.eigh(np.matrix(self.moi))
    self.axes = np.array(np.transpose(self.axes)[::-1])

  # determine molecule type based on principal moments of inertia
  def get_moltype(self):
    haszero = are_same(self.pmoms[0], 0.0, thresh, mom_min)
    same12  = are_same(self.pmoms[0], self.pmoms[1], thresh, mom_min)
    same23  = are_same(self.pmoms[1], self.pmoms[2], thresh, mom_min)
    allsame = same12 * same23
    allzero = allsame * haszero
    if   (allzero): self.moltype = 'monatomic'
    elif (haszero): self.moltype = 'linear'
    elif (allsame): self.moltype = 'a spherical top'
    elif  (same12): self.moltype = 'an oblate symmetric top'
    elif  (same23): self.moltype = 'a prolate symmetric top'
    else          : self.moltype = 'an asymmetric top'

  # calculate principal moments of inertia (eigenvalues of tensor)
  def get_axes(self, rerank):
    self.get_com()
    self.get_moi(self.com, global_axes)
    self.get_inertial_axes()
    if (rerank): self.get_moltype()
    self.get_local_coords(self.com, self.axes)
    self.align_coords(rerank)
    if (rerank): self.get_orient_params()

    #print 'rerank = %s' % (rerank)
    #print 'moltype = %s' % (self.moltype)
    #print 'atom rankings'
    #for i in range(self.n_atoms):
    #  print '  atom %02i rank:' % (i+1),
    #  for j in range(len(self.atoms[i].rank)):
    #    print ' %i: %2i,' % (j+1, self.atoms[i].rank[j] + 1),
    #  print '\n',
    #self.print_local_geom(self.com, self.axes, 'local geometry')
    #print 'global geometry'
    #for i in range(self.n_atoms):
    #  print ' %-2s' % (self.atoms[i].attype),
    #  for j in range(3):
    #    print ' %12.6f' % (self.atoms[i].coords[j]),
    #  print '\n',
    #print 'axes'
    #for i in range(3):
    #  for j in range(3):
    #    print ' %9.6f' % (self.axes[i][j]),
    #  print '\n',
    #print '\n',

  # get orientation angle default parameters based on molecular type
  def get_orient_params(self):
    if   (self.moltype == 'monatomic'):
      self.set_orient_params(self.axes, 0.0, 0.0, 0.0)
    elif (self.moltype == 'linear'):
      self.set_orient_params(self.axes, 0.0, 0.0, 3.0*self.k_ref)
    else:
      self.set_orient_params(self.axes, self.k_ref, self.k_ref, self.k_ref)

  # set orientation angle default parameters (spring constants and angles)
  def set_orient_params(self, ref_axes, kx, ky, kz):
    self.orient_springs = np.array([kx, ky, kz])
    self.orient_ref_axes = np.array(ref_axes)

  # calculate energy of molecular orientation relative to reference state
  def get_orient_energy(self):
    self.orient_energy = 0.0
    self.orient_diag = np.zeros(3)
    for i in range(3):
      self.orient_diag[i] = get_udp(self.axes[i], self.orient_ref_axes[i])
    for i in range(3):
      kval = self.orient_springs[i]
      dval = self.orient_diag[i]
      self.orient_energy += 0.25 * kval * (1.0 - dval)

  # calculate gradient of orientation energy with respect to global coordinates
  def get_orient_gradient(self):
    self.orient_gradient = np.array(np.zeros((self.n_atoms, 3)))
    self.get_global_coords()
    for i in range(self.n_atoms):
      for j in range(3):
        q = self.global_coords[i][j]
        qp = q + 0.5*fin_diff
        qm = q - 0.5*fin_diff
        self.atoms[i].coords[j] = qp
        self.get_axes(False)
        self.get_orient_energy()
        ep = self.orient_energy
        self.atoms[i].coords[j] = qm
        self.get_axes(False)
        self.get_orient_energy()
        em = self.orient_energy
        self.atoms[i].coords[j] = q
        self.orient_gradient[i][j] = (ep - em) / fin_diff
    self.get_axes(False)

  # rotate axes by a defined global rotation matrix
  def rotate_axes(self, matrix):
    self.axes = np.array(np.dot(self.axes, np.transpose(matrix)))

  # exchange two axes for one another in a 90-degree rotation
  def exchange_coords(self, ax1, ax2):
    temp = 1.0 * self.axes[ax1]
    self.axes[ax1] = 1.0 * self.axes[ax2]
    self.axes[ax2] = -1.0 * temp
    self.get_local_coords(self.com, self.axes)

  # calculate coordinates for unit vector local axes from center of mass
  def get_local_coords(self, point, axes):
    self.local_coords = np.zeros((self.n_atoms, 3))
    for i in range(self.n_atoms):
      r_ip = get_r12(point, self.atoms[i].coords)
      u_ip = get_u12(point, self.atoms[i].coords)
      for j in range(3):
        self.local_coords[i][j] += r_ip * get_udp(axes[j], u_ip)

  # return an nx3 array of global atomic coordinates
  def get_global_coords(self):
    self.global_coords = np.array(np.zeros((self.n_atoms, 3)))
    for i in range(self.n_atoms):
      for j in range(3):
        self.global_coords[i][j] = self.atoms[i].coords[j]

  # determine distance of atoms from a point
  def get_point_dist(self, point, point_name):
    for i in range(self.n_atoms):
      point_dist = get_r12(self.atoms[i].coords, point)
      self.atoms[i].dists[point_name] = point_dist

  # determine distance of atoms from an axis
  def get_axis_dist(self, point, vector, axis_name):
    for i in range(self.n_atoms):
      r_pi = get_r12(point, self.atoms[i].coords)
      u_pi = get_u12(point, self.atoms[i].coords)
      u_ax = get_u12([0.0, 0.0, 0.0], vector)
      dp = get_udp(u_ax, u_pi)
      axis_dist = r_pi * math.sqrt(1 - dp**2)
      self.atoms[i].dists[axis_name] = axis_dist

  # determine distance of atoms from a plane
  def get_plane_dist(self, point, normal, plane_name, signed):
    for i in range(self.n_atoms):
      r_pi = get_r12(point, self.atoms[i].coords)
      u_pi = get_u12(point, self.atoms[i].coords)
      u_nm = get_u12([0.0, 0.0, 0.0], normal)
      dp = get_udp(u_nm, u_pi)
      plane_dist = r_pi * dp
      if       (signed): self.atoms[i].dists[plane_name] =     plane_dist
      elif (not signed): self.atoms[i].dists[plane_name] = abs(plane_dist)

  # set values which are indistinguishable from zero to infinity
  def set_nozero(self, comp):
    for i in range(self.n_atoms):
      if (are_same(self.atoms[i].dists[comp], 0.0, thresh, cmin)):
        self.atoms[i].dists[comp] = sys.float_info.max

  # set values which are indistinguishable from zero to zero
  def set_samezero(self, comp):
    for i in range(self.n_atoms):
      if (are_same(self.atoms[i].dists[comp], 0.0, thresh, cmin)):
        self.atoms[i].dists[comp] = 0.0

  # reset all rank values of atoms to atomic order
  def reset_all_rank(self):
    for k in range(len(self.atoms[0].rank)):
      self.reset_rank(k)

  # reset kth rank value of atoms to atomic order
  def reset_rank(self, k):
    for i in range(self.n_atoms):
      self.atoms[i].rank[k] = i

  # make kth rank of i higher priority (lower) than j if ival is greater than jval
  def order_rank(self, i, j, ival, jval, k):
    r1 = self.atoms[i].rank[k]
    r2 = self.atoms[j].rank[k]
    ranked = False
    if (not are_same(ival, jval, thresh, cmin)):
      self.atoms[i].rank[k] = min(r1, r2) if (ival >= jval) else max(r1, r2)
      self.atoms[j].rank[k] = max(r1, r2) if (ival >= jval) else min(r1, r2)
      ranked = True
    return ranked

  # get atomic index with given kth rank
  def find_rank_index(self, n, k):
    for i in range(self.n_atoms):
      if (self.atoms[i].rank[k] == n):
        return i

  # get index of minimum value atom of a criterion
  def find_min_index(self, comp):
    minval = sys.float_info.max
    minind = 0
    for i in range(self.n_atoms):
      dist = self.atoms[i].dists[comp]
      if (dist < minval and not are_same(dist, minval, thresh, cmin)):
        minval = dist
        minind = i
    return minind

  # rank atoms by spherical top criteria around origin
  def rank_sphertop(self):
    self.reset_all_rank()
    comp1 = 'Center-of-mass'
    self.get_point_dist(self.com, comp1)
    self.set_samezero(comp1)
    for k in range(3):
      for a in range(self.n_atoms):
        for b in range(a+1, self.n_atoms):
          i, j = self.find_rank_index(a, k), self.find_rank_index(b, k)
          at1, at2 = self.atoms[i], self.atoms[j]
          if   (k == 0): ival, jval = at1.dists[comp1], at2.dists[comp1]
          elif (k == 1): ival, jval = at1.charge, at2.charge
          elif (k == 2): ival, jval = i, j
          self.order_rank(i, j, ival, jval, k)

  # rank atoms by symmetric top criteria around chosen axis
  def rank_symtop(self, axes, ax1, ax2):
    self.reset_all_rank()
    ax3 = 3 - ax1 - ax2
    comp1 = '%s%s-plane' % (xyz[min(ax2, ax3)], xyz[max(ax2, ax3)])
    comp2 = '%s-axis' % (xyz[ax1])
    self.get_plane_dist(self.com, axes[ax1], comp1, False)
    self.get_axis_dist( self.com, axes[ax1], comp2)
    self.set_samezero(comp1)
    self.set_samezero(comp2)
    for k in range(4):
      for a in range(self.n_atoms):
        for b in range(a+1, self.n_atoms):
          i, j = self.find_rank_index(a, k), self.find_rank_index(b, k)
          at1, at2 = self.atoms[i], self.atoms[j]
          if   (k == 0): ival, jval = at1.dists[comp1], at2.dists[comp1]
          elif (k == 1): ival, jval = at1.dists[comp2], at2.dists[comp2]
          elif (k == 2): ival, jval = at1.charge, at2.charge
          elif (k == 3): ival, jval = i, j
          self.order_rank(i, j, ival, jval, k)

  # rank atoms by asymmetric top criteria around chosen axes
  def rank_asymtop(self, axes, ax1, ax2):
    self.reset_all_rank()
    ax3 = 3 - ax1 - ax2
    comp1 = 'Center-of-mass'
    comp2 = '%s%s-plane' % (xyz[ax3], xyz[ax2])
    comp3 = '%s%s-plane' % (xyz[ax1], xyz[ax3])
    comp4 = '%s%s-plane' % (xyz[ax2], xyz[ax1])
    self.get_point_dist(self.com, comp1)
    self.get_plane_dist(self.com, axes[ax1], comp2, False)
    self.get_plane_dist(self.com, axes[ax2], comp3, False)
    self.get_plane_dist(self.com, axes[ax3], comp4, False)
    for k in range(5):
      for a in range(self.n_atoms):
        for b in range(a+1, self.n_atoms):
          i, j = self.find_rank_index(a, k), self.find_rank_index(b, k)
          at1, at2 = self.atoms[i], self.atoms[j]
          if   (k == 0): ival, jval = at1.dists[comp1], at2.dists[comp1]
          elif (k == 1): ival, jval = at1.dists[comp2], at2.dists[comp2]
          elif (k == 2): ival, jval = at1.dists[comp3], at2.dists[comp3]
          elif (k == 3): ival, jval = at1.dists[comp4], at2.dists[comp4]
          elif (k == 4): ival, jval = i, j
          self.order_rank(i, j,  ival,  jval, k)

  # rank atoms by linear criteria around molecular axis
  def rank_linear(self, axes, ax1, ax2):
    self.reset_all_rank()
    ax3 = 3 - ax1 - ax2
    comp1 = '%s%s-plane' % (xyz[min(ax2,ax3)], xyz[max(ax2,ax3)])
    self.get_plane_dist(self.com, axes[ax1], comp1, False)
    self.set_samezero(comp1)
    for k in range(3):
      for a in range(self.n_atoms):
        for b in range(a+1, self.n_atoms):
          i, j = self.find_rank_index(a, k), self.find_rank_index(b, k)
          at1, at2 = self.atoms[i], self.atoms[j]
          if   (k == 0): ival, jval = at1.dists[comp1], at2.dists[comp1]
          elif (k == 1): ival, jval = at1.charge, at2.charge
          elif (k == 2): ival, jval = i, j
          self.order_rank(i, j,  ival,  jval, k)

  # rotate spherical top to make key atom lie in +z-axis
  def align_sphertop(self, ax1, ax2, rerank):
    if (rerank): self.rank_sphertop()
    for i in range(1, self.n_atoms):
      at1 = self.find_rank_index(0, 0)
      at2 = self.find_rank_index(i, 0)
      #print 'at1 = %i, at2 = %i' % (at1+1, at2+1)
      u_at1 = get_u12(self.com, self.atoms[at1].coords)
      u_at2 = get_u12(self.com, self.atoms[at2].coords)
      if (abs(get_udp(u_at1, u_at2)) < 0.60): break
    self.axes[xyz['Z']] = u_at1
    self.axes[xyz['X']] = get_ucp(u_at2, u_at1)
    self.axes[xyz['Y']] = get_ucp(self.axes[xyz['Z']], self.axes[xyz['X']])
    self.get_local_coords(self.com, self.axes)

  # rotate symmetric top to make key atom lie in yz-plane in +y-axis
  def align_symtop(self, axes, ax1, ax2, rerank):
    if (rerank): self.rank_symtop(axes, ax1, ax2)
    ax3 = 3 - ax1 - ax2
    comp1 = '%s%s-plane' % (xyz[min(ax2,ax3)], xyz[max(ax2,ax3)])
    comp2 = '%s-axis' % (xyz[ax1])
    at1 = self.find_rank_index(0, 0)
    at2 = self.find_rank_index(0, 1)
    if (not are_same(self.atoms[at1].dists[comp1], 0.0, thresh, cmin)):
      z_scale = 1.0 if (self.local_coords[at1][ax1] > 0.0) else -1.0
      u_at2 = get_u12(self.com, self.atoms[at2].coords)
      self.axes[xyz['Z']] *= z_scale
      self.axes[xyz['X']] = get_ucp(u_at2, self.axes[xyz['Z']])
      self.axes[xyz['Y']] = get_ucp(self.axes[xyz['Z']], self.axes[xyz['X']])
      self.get_local_coords(self.com, self.axes)
    else:
      for i in range(1, self.n_atoms):
        at1 = self.find_rank_index(0, 1)
        at2 = self.find_rank_index(i, 1)
        u_at1 = get_u12(self.com, self.atoms[at1].coords)
        u_at2 = get_u12(self.com, self.atoms[at2].coords)
        if (abs(get_udp(u_at1, u_at2)) < 0.60): break
      self.axes[xyz['Y']] = u_at1
      self.axes[xyz['Z']] = get_ucp(u_at2, u_at1)
      self.axes[xyz['X']] = get_ucp(self.axes[xyz['Y']], self.axes[xyz['Z']])
      self.get_local_coords(self.com, self.axes)

  # rotate asymmetric top to make key atom lie in +y+z quadrant
  def align_asymtop(self, axes, ax1, ax2, rerank):
    if (rerank): self.rank_asymtop(axes, ax1, ax2)
    ax3 = 3 - ax1 - ax2
    comp1 = '%s%s-plane' % (xyz[ax3], xyz[ax2])
    comp2 = '%s%s-plane' % (xyz[ax1], xyz[ax3])
    at1 = self.find_rank_index(0, 1)
    at2 = self.find_rank_index(0, 2)
    #print 'at1 = %i, at2 = %i' % (at1+1, at2+1)
    z_scale = 1.0 if (self.local_coords[at1][ax1] >= 0.0) else -1.0
    y_scale = 1.0 if (self.local_coords[at2][ax2] >= 0.0) else -1.0
    #print 'zsc = %i, ysc = %i' % (z_scale, y_scale)
    #print 'zval = %.6f, yval = %.6f' % (self.local_coords[at1][ax1], self.local_coords[at2][ax2])
    zl = z_scale * self.axes[xyz['Z']]
    yl = y_scale * self.axes[xyz['Y']]
    xl = get_ucp(yl, zl)
    self.axes = np.array([xl, yl, zl])
    self.get_local_coords(self.com, self.axes)

  # rotate linear molecule to make key atom lie on +z axis
  def align_linear(self, axes, ax1, ax2, rerank):
    if (rerank): self.rank_linear(axes, ax1, ax2)
    ax3 = 3 - ax1 - ax2
    comp1 = '%s%s-plane' % (xyz[ax3], xyz[ax2])
    at1 = self.find_rank_index(0, 0)
    zl = get_u12(self.com, self.atoms[at1].coords)
    xl = get_ucp(global_axes[xyz['Z']], zl)
    if (la.norm(xl) == 0.0): xl = global_axes[xyz['X']]
    yl = get_ucp(zl, xl)
    self.axes = np.array([xl, yl, zl])
    self.get_local_coords(self.com, self.axes)

  # rotate atom to make local axes global axes
  def align_monatomic(self):
    self.axes = np.array(global_axes)
    self.get_local_coords(self.com, self.axes)

  # align molecule based on molecular type
  def align_coords(self, rerank):
    if (self.moltype == 'an asymmetric top'):
      self.align_asymtop(self.axes, xyz['Z'], xyz['Y'], rerank)
    elif (self.moltype == 'an oblate symmetric top'):
      self.exchange_coords(xyz['X'], xyz['Z'])
      self.align_symtop(self.axes, xyz['Z'], xyz['Y'], rerank)
    elif (self.moltype == 'a prolate symmetric top'):
      self.align_symtop(self.axes, xyz['Z'], xyz['Y'], rerank)
    elif (self.moltype == 'a spherical top'):
      self.align_sphertop(xyz['Z'], xyz['Y'], rerank)
    elif (self.moltype == 'linear'):
      self.align_linear(self.axes, xyz['Z'], xyz['Y'], rerank)
    elif (self.moltype == 'monatomic'):
      self.align_monatomic()

  # print local geometry to screen
  def print_local_geom(self, point, axes, comment):
    self.get_local_coords(point, axes)
    print '%i\n%s\n' % (self.n_atoms, comment),
    for i in range(self.n_atoms):
      print '%-2s' % (self.atoms[i].attype),
      for j in range(3):
        print ' %12.6f' % (self.local_coords[i][j]),
      print '\n',

  # print orientation energy gradient
  def print_orient_gradient(self, comment):
    print '\n%s\n' % (comment),
    for i in range(self.n_atoms):
      print '%-2s' % (self.atoms[i].attype),
      for j in range(3):
        print ' %12.6f' % (self.orient_gradient[i][j]),
      print '\n',

