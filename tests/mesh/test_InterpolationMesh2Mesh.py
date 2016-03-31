import numpy as np
import unittest
from SimPEG.Utils import mkvc
from SimPEG import Mesh, Tests
import unittest

test1D = False
test2D = True
test3D = False

call1 = lambda fun, xyz: fun(xyz)
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, -1])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cart_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cart_row3 = lambda g, xfun, yfun, zfun: np.c_[call3(xfun, g), call3(yfun, g), call3(zfun, g)]
cartF2 = lambda M, fx, fy: np.vstack((cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy)))
cartF2Cyl = lambda M, fx, fy: np.vstack((cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFz, fx, fy)))
cartE2 = lambda M, ex, ey: np.vstack((cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey)))
cartE2Cyl = lambda M, ex, ey: cart_row2(M.gridEy, ex, ey)
cartF3 = lambda M, fx, fy, fz: np.vstack((cart_row3(M.gridFx, fx, fy, fz), cart_row3(M.gridFy, fx, fy, fz), cart_row3(M.gridFz, fx, fy, fz)))
cartE3 = lambda M, ex, ey, ez: np.vstack((cart_row3(M.gridEx, ex, ey, ez), cart_row3(M.gridEy, ex, ey, ez), cart_row3(M.gridEz, ex, ey, ez)))

TOL = 1e-7

if test1D:
    class TestInterpolationMesh2Mesh_Tensor1D(Tests.OrderTest):

        name = 'Mesh2Mesh Tensor1D'
        meshSizes = [8, 16, 32]
        meshTypes = ['uniformTensorMesh']
        meshDimension = 1

        def getError(self):
            funX = lambda x: np.cos(2*np.pi*x)

            mesh2, _ = self.makeMesh(self.M.nC-1, meshType=self._meshType, meshDimension=self.meshDimension )
            ana = call1(funX, getattr(mesh2, 'grid%s'%self.type))

            v = call1(funX, getattr(self.M, 'grid%s'%self.type))
            P = self.M.getInterpolationMatMesh2Mesh(mesh2, locType=self.type)
            num = P*v

            return np.linalg.norm((num - ana), np.inf)

        def test_orderCC_1D(self):
            self.type = 'CC'
            self.name = 'Mesh2Mesh Tensor1D: CC'
            self.orderTest()

        def test_orderN_1D(self):
            self.type = 'N'
            self.name = 'Mesh2Mesh Tensor1D: N'
            self.orderTest()

        def test_orderEx_1D(self):
            self.type = 'Ex'
            self.name = 'Mesh2Mesh Tensor1D: Ex'
            self.orderTest()

        def test_orderFx_1D(self):
            self.type = 'Fx'
            self.name = 'Mesh2Mesh Tensor1D: Fx'
            self.orderTest()

if test2D:
    class TestInterpolationMesh2Mesh_Tensor2D(Tests.OrderTest):

        name = 'Mesh2Mesh Tensor2D'
        meshSizes = [4, 8, 16]
        meshTypes = ['uniformTensorMesh']
        meshDimension = 2

        def getError(self):
            funX = lambda x, y: np.cos(2*np.pi*y)
            funY = lambda x, y: np.cos(2*np.pi*x)

            mesh2, _ = self.makeMesh(self.M.nC-1, meshType=self._meshType, meshDimension=self.meshDimension )

            if 'x' in self.type:
                ana = call2(funX, getattr(mesh2, 'grid%s'%self.type))
            elif 'y' in self.type:
                ana = call2(funY, getattr(mesh2, 'grid%s'%self.type))
            elif 'F' in self.type:
                ana = cartF2(mesh2, funX, funY)
                ana = mesh2.projectFaceVector(ana)
            elif 'E' in self.type:
                ana = cartE2(mesh2, funX, funY)
                ana = mesh2.projectEdgeVector(ana)
            else:
                ana = call2(funX, getattr(mesh2, 'grid%s'%self.type))


            if 'F' in self.type:
                v = cartF2(self.M, funX, funY)
                if 'x' in self.type or 'y' in self.type:
                    v = self.M.projectFaceVector(v)
                else:
                    v = mkvc(v)
            elif 'E' in self.type:
                v = cartE2(self.M, funX, funY)
                if 'x' in self.type or 'y' in self.type:
                    v = self.M.projectEdgeVector(v)
                else:
                    v = mkvc(v)
            elif 'CC' == self.type:
                v = call2(funX, self.M.gridCC)
            elif 'N' == self.type:
                v = call2(funX, self.M.gridN)

            P = self.M.getInterpolationMatMesh2Mesh(mesh2, locType=self.type)
            # print P.shape, v.shape
            num = P*v

            return np.linalg.norm((num - ana), np.inf)

        def test_orderCC_2D(self):
            self.type = 'CC'
            self.name = 'Mesh2Mesh Tensor2D: CC'
            self.orderTest()

        def test_orderN_2D(self):
            self.type = 'N'
            self.name = 'Mesh2Mesh Tensor2D: N'
            self.orderTest()

        def test_orderE_2D(self):
            self.type = 'E'
            self.name = 'Mesh2Mesh Tensor2D: E'
            self.orderTest()

        def test_orderEx_2D(self):
            self.type = 'Ex'
            self.name = 'Mesh2Mesh Tensor2D: Ex'
            self.orderTest()

        def test_orderEy_2D(self):
            self.type = 'Ey'
            self.name = 'Mesh2Mesh Tensor2D: Ey'
            self.orderTest()

        def test_orderF_2D(self):
            self.type = 'F'
            self.name = 'Mesh2Mesh Tensor2D: F'
            self.orderTest()

        def test_orderFx_2D(self):
            self.type = 'Fx'
            self.name = 'Mesh2Mesh Tensor2D: Fx'
            self.orderTest()

        def test_orderFy_2D(self):
            self.type = 'Fy'
            self.name = 'Mesh2Mesh Tensor2D: Fy'
            self.orderTest()

    class TestInterpolationMesh2Mesh_Cyl(Tests.OrderTest):

        name = 'Mesh2Mesh Cyl'
        meshSizes = [4, 8, 16]
        meshTypes = ['uniformCylMesh']
        meshDimension = 2

        def getError(self):
            funX = lambda x, y: np.cos(2*np.pi*y)
            funY = lambda x, y: np.cos(2*np.pi*x)

            mesh2, _ = self.makeMesh(self.M.nC-1, meshType=self._meshType, meshDimension=self.meshDimension )

            if 'x' in self.type:
                ana = call2(funX, getattr(mesh2, 'grid%s'%self.type))
            elif 'y' in self.type:
                ana = call2(funY, getattr(mesh2, 'grid%s'%self.type))
            elif 'z' in self.type:
                ana = call2(funY, getattr(mesh2, 'grid%s'%self.type))
            elif 'F' in self.type:
                ana = cartF2Cyl(mesh2, funX, funY)
                ana = np.c_[ana[:,0], np.zeros_like(ana[:,0]), ana[:,1]]
                ana = mesh2.projectFaceVector(ana)
            elif 'E' in self.type:
                ana = cartE2Cyl(mesh2, funX, funY)
                ana = np.c_[np.zeros_like(ana[:,1]),ana[:,1],np.zeros_like(ana[:,1])]
                ana = mesh2.projectEdgeVector(ana)
            else:
                ana = call2(funX, getattr(mesh2, 'grid%s'%self.type))


            if 'F' in self.type:
                v = cartF2Cyl(self.M, funX, funY)
                v = np.c_[v[:,0], np.zeros_like(v[:,0]),v[:,1]]
                if 'x' in self.type or 'z' in self.type:
                    v = self.M.projectFaceVector(v)
                else:
                    v = np.c_[v[:,0], v[:,2]]
                    v = mkvc(v)
            elif 'E' in self.type:
                v = cartE2Cyl(self.M, funX, funY)
                v = np.c_[np.zeros_like(v[:,1]), v[:,1],np.zeros_like(v[:,1])]
                v = self.M.projectEdgeVector(v)

            elif 'CC' == self.type:
                v = call2(funX, self.M.gridCC)
            elif 'N' == self.type:
                v = call2(funX, self.M.gridN)

            P = self.M.getInterpolationMatMesh2Mesh(mesh2, locType=self.type)
            print P.shape, v.shape
            num = P*v

            return np.linalg.norm((num - ana), np.inf)

        def test_orderCC_Cyl(self):
            self.type = 'CC'
            self.name = 'Mesh2Mesh Tensor2D: CC'
            self.orderTest()

        def test_orderN_Cyl(self):
            self.type = 'N'
            self.name = 'Mesh2Mesh Tensor2D: N'
            self.orderTest()

        def test_orderE_Cyl(self):
            self.type = 'E'
            self.name = 'Mesh2Mesh Tensor2D: E'
            self.orderTest()

        def test_orderEy_Cyl(self):
            self.type = 'Ey'
            self.name = 'Mesh2Mesh Tensor2D: Ey'
            self.orderTest()

        def test_orderF_Cyl(self):
            self.type = 'F'
            self.name = 'Mesh2Mesh Tensor2D: F'
            self.orderTest()

        def test_orderFx_Cyl(self):
            self.type = 'Fx'
            self.name = 'Mesh2Mesh Tensor2D: Fx'
            self.orderTest()

        def test_orderFz_Cyl(self):
            self.type = 'Fz'
            self.name = 'Mesh2Mesh Tensor2D: Fz'
            self.orderTest()

if test3D:
    class TestInterpolationMesh2Mesh_Tensor3D(Tests.OrderTest):

        name = 'Mesh2Mesh Tensor3D'
        meshSizes = [4, 8, 16]
        meshTypes = ['uniformTensorMesh']
        meshDimension = 3

        def getError(self):
            funX = lambda x, y, z: np.cos(2*np.pi*y)
            funY = lambda x, y, z: np.cos(2*np.pi*z)
            funZ = lambda x, y, z: np.cos(2*np.pi*x)

            mesh2, _ = self.makeMesh(self.M.nC-1, meshType=self._meshType, meshDimension=self.meshDimension )

            if 'x' in self.type:
                ana = call3(funX, getattr(mesh2, 'grid%s'%self.type))
            elif 'y' in self.type:
                ana = call3(funY, getattr(mesh2, 'grid%s'%self.type))
            elif 'z' in self.type:
                ana = call3(funZ, getattr(mesh2, 'grid%s'%self.type))
            elif 'F' in self.type:
                ana = cartF3(mesh2, funX, funY, funZ)
                ana = mesh2.projectFaceVector(ana)
            elif 'E' in self.type:
                ana = cartE3(mesh2, funX, funY, funZ)
                ana = mesh2.projectFaceVector(ana)
            else:
                ana = call3(funX, getattr(mesh2, 'grid%s'%self.type))


            if 'F' in self.type:
                v = cartF3(self.M, funX, funY, funZ)
                if 'x' in self.type or 'y' in self.type or 'z' in self.type:
                    v = self.M.projectFaceVector(v)
                else:
                    v = mkvc(v)
            elif 'E' in self.type:
                v = cartE3(self.M, funX, funY, funZ)
                if 'x' in self.type or 'y' in self.type or 'z' in self.type:
                    v = self.M.projectFaceVector(v)
                else:
                    v = mkvc(v)
            elif 'CC' == self.type:
                v = call3(funX, self.M.gridCC)
            elif 'N' == self.type:
                v = call3(funX, self.M.gridN)

            P = self.M.getInterpolationMatMesh2Mesh(mesh2, locType=self.type)
            # print P.shape, v.shape
            num = P*v

            return np.linalg.norm((num - ana), np.inf)

        def test_orderCC_3D(self):
            self.type = 'CC'
            self.name = 'Mesh2Mesh Tensor3D: CC'
            self.orderTest()

        def test_orderN_3D(self):
            self.type = 'N'
            self.name = 'Mesh2Mesh Tensor3D: N'
            self.orderTest()

        def test_orderE_3D(self):
            self.type = 'E'
            self.name = 'Mesh2Mesh Tensor3D: E'
            self.orderTest()

        def test_orderEx_3D(self):
            self.type = 'Ex'
            self.name = 'Mesh2Mesh Tensor3D: Ex'
            self.orderTest()

        def test_orderEy_3D(self):
            self.type = 'Ey'
            self.name = 'Mesh2Mesh Tensor3D: Ey'
            self.orderTest()

        def test_orderEz_3D(self):
            self.type = 'Ez'
            self.name = 'Mesh2Mesh Tensor3D: Ez'
            self.orderTest()

        def test_orderF_3D(self):
            self.type = 'F'
            self.name = 'Mesh2Mesh Tensor3D: F'
            self.orderTest()

        def test_orderFx_3D(self):
            self.type = 'Fx'
            self.name = 'Mesh2Mesh Tensor3D: Fx'
            self.orderTest()

        def test_orderFy_3D(self):
            self.type = 'Fy'
            self.name = 'Mesh2Mesh Tensor3D: Fy'
            self.orderTest()

        def test_orderFz_3D(self):
            self.type = 'Fz'
            self.name = 'Mesh2Mesh Tensor3D: Fz'
            self.orderTest()




if __name__ == '__main__':
    unittest.main()
