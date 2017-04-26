//
// Quick start:
//
// - Compile the code with
//     g++ -O2 code-2016.cpp -o Cavity
//
// - Run the code
//     ./Cavity 12 1e-4 400 driven-cavity
//
// - Open Paraview and load all the output...vtk files in one rush.
//
// - Select glyphs from the toolbox. Press apply and press the play button
//   (arrow top right).
//



#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vtkCylinderSource.h>
#include <vtkPolyData.h>
#include <vtkCellData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkUnstructuredGrid.h>
#include <vtkHexahedron.h>
#include <vtkDataSetMapper.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridGeometryFilter.h>
#include <vtkArrowSource.h>
#include <vtkGlyph3D.h>
#include <vtkImageData.h>
#include <vtkDoubleArray.h>
#include <vtkDataSetAttributes.h>
#include <vtkAssignAttribute.h>
#include <vtkPointData.h>
#include <vtkPolyDataWriter.h>
#include <vtkDataSet.h>
#include <vtkLookupTable.h>
#include <thread>
#include <vtkInteractorStyleTrackballCamera.h>

/**
* Number of cell we have per axis
*/
int numberOfCellsPerAxisX;
int numberOfCellsPerAxisY;
int numberOfCellsPerAxisZ;

/**
* Velocity of fluid along x,y and z direction
*/
double* ux;
double* uy;
double* uz;

/**
* Helper along x,y and z direction
*/
double* Fx;
double* Fy;
double* Fz;

/**
* Pressure in the cell
*/
double* p;
double* rhs;

double timeStepSize;

double ReynoldsNumber = 0;

const int MaxComputePIterations = 10000;
const int MinAverageComputePIterations = 100;
const int IterationsBeforeTimeStepSizeIsAltered = 64;
const double ChangeOfTimeStepSize = 0.1;
const double PPESolverThreshold = 1e-4;
const double TerminalTime = 10.0;

double maxPressure;
double minPressure;
vtkSmartPointer<vtkLookupTable> colorLookupTable;
vtkSmartPointer<vtkDataSetMapper> pressureMapper;

/**
* Switch on to have a couple of security checks
*/
//#define CheckVariableValues


/**
* This is a small macro that I use to ensure that something is valid. There's
* a more sophisticated version of this now in the C++ standard (and there are
* many libs that have way more mature assertion functions), but it does the
* job. Please comment it out if you do runtime measurements - you can do so by
* commenting out the define statement above.
*/
void assertion(bool condition, int line) {
#ifdef CheckVariableValues
	if (!condition) {
		std::cerr << "assertion failed in line " << line << std::endl;
		exit(-1);
	}
#endif
}

/**
* Maps the three coordinates onto one cell index.
*/
int getCellIndex(int ix, int iy, int iz) {
	assertion(ix >= 0, __LINE__);
	assertion(ix<numberOfCellsPerAxisX + 2, __LINE__);
	assertion(iy >= 0, __LINE__);
	assertion(iy<numberOfCellsPerAxisY + 2, __LINE__);
	assertion(iz >= 0, __LINE__);
	assertion(iz<numberOfCellsPerAxisZ + 2, __LINE__);
	return ix + iy*(numberOfCellsPerAxisX + 2) + iz*(numberOfCellsPerAxisX + 2)*(numberOfCellsPerAxisY + 2);
}

/**
* Maps the three coordinates onto one vertex index.
* Please note that we hold only the inner and boundary vertices.
*/
int getVertexIndex(int ix, int iy, int iz) {
	assertion(ix >= 0, __LINE__);
	assertion(ix<numberOfCellsPerAxisX + 1, __LINE__);
	assertion(iy >= 0, __LINE__);
	assertion(iy<numberOfCellsPerAxisY + 1, __LINE__);
	assertion(iz >= 0, __LINE__);
	assertion(iz<numberOfCellsPerAxisZ + 1, __LINE__);
	return ix + iy*(numberOfCellsPerAxisX + 1) + iz*(numberOfCellsPerAxisX + 1)*(numberOfCellsPerAxisY + 1);
}


/**
* Gives you the face with the number ix,iy,iz.
*
* Takes into account that there's one more face in X direction than numberOfCellsPerAxisX.
*/
int getFaceIndexX(int ix, int iy, int iz) {
	assertion(ix >= 0, __LINE__);
	assertion(ix<numberOfCellsPerAxisX + 3, __LINE__);
	assertion(iy >= 0, __LINE__);
	assertion(iy<numberOfCellsPerAxisY + 2, __LINE__);
	assertion(iz >= 0, __LINE__);
	assertion(iz<numberOfCellsPerAxisZ + 2, __LINE__);
	return ix + iy*(numberOfCellsPerAxisX + 3) + iz*(numberOfCellsPerAxisX + 3)*(numberOfCellsPerAxisY + 2);
}


int getFaceIndexY(int ix, int iy, int iz) {
	assertion(ix >= 0, __LINE__);
	assertion(ix<numberOfCellsPerAxisX + 2, __LINE__);
	assertion(iy >= 0, __LINE__);
	assertion(iy<numberOfCellsPerAxisY + 3, __LINE__);
	assertion(iz >= 0, __LINE__);
	assertion(iz<numberOfCellsPerAxisZ + 2, __LINE__);
	return ix + iy*(numberOfCellsPerAxisX + 2) + iz*(numberOfCellsPerAxisX + 2)*(numberOfCellsPerAxisY + 2);
}


int getFaceIndexZ(int ix, int iy, int iz) {
	assertion(ix >= 0, __LINE__);
	assertion(ix<numberOfCellsPerAxisX + 2, __LINE__);
	assertion(iy >= 0, __LINE__);
	assertion(iy<numberOfCellsPerAxisY + 2, __LINE__);
	assertion(iz >= 0, __LINE__);
	assertion(iz<numberOfCellsPerAxisZ + 3, __LINE__);
	return ix + iy*(numberOfCellsPerAxisX + 2) + iz*(numberOfCellsPerAxisX + 2)*(numberOfCellsPerAxisY + 2);
}


/**
* We use always numberOfCellsPerAxisX=numberOfCellsPerAxisY and we make Z 5
* times bigger, i.e. we always work with cubes. We also assume that the whole
* setup always has exactly the height 1.
*/
double getH() {
	return 1.0 / static_cast<double>(numberOfCellsPerAxisY);
}


/**
* There are two types of errors/bugs that really hunt us in these codes: We
* either have programmed something wrong (use wrong indices) or we make a
* tiny little error in one of the computations. The first type of errors is
* covered by assertions. The latter type is realised in this routine, where
* we do some consistency checks.
*/
void validateThatEntriesAreBounded(const std::string&  callingRoutine) {
#ifdef CheckVariableValues
	for (int ix = 0; ix<numberOfCellsPerAxisX + 2; ix++)
		for (int iy = 0; iy<numberOfCellsPerAxisY + 2; iy++)
			for (int iz = 0; iz<numberOfCellsPerAxisZ + 2; iz++) {
				if (std::abs(p[getCellIndex(ix, iy, iz)])>1e10) {
					std::cerr << "error in routine " + callingRoutine + " in p[" << ix << "," << iy << "," << iz << "]" << std::endl;
					exit(-1);
				}
			}

	for (int ix = 0; ix<numberOfCellsPerAxisX + 3; ix++)
		for (int iy = 0; iy<numberOfCellsPerAxisY + 2; iy++)
			for (int iz = 0; iz<numberOfCellsPerAxisZ + 2; iz++) {
				if (std::abs(ux[getFaceIndexX(ix, iy, iz)])>1e10) {
					std::cerr << "error in routine " + callingRoutine + " in ux[" << ix << "," << iy << "," << iz << "]" << std::endl;
					exit(-1);
				}
				if (std::abs(Fx[getFaceIndexX(ix, iy, iz)])>1e10) {
					std::cerr << "error in routine " + callingRoutine + " in Fx[" << ix << "," << iy << "," << iz << "]" << std::endl;
					exit(-1);
				}
			}

	for (int ix = 0; ix<numberOfCellsPerAxisX + 2; ix++)
		for (int iy = 0; iy<numberOfCellsPerAxisY + 3; iy++)
			for (int iz = 0; iz<numberOfCellsPerAxisZ + 2; iz++) {
				if (std::abs(uy[getFaceIndexY(ix, iy, iz)])>1e10) {
					std::cerr << "error in routine " + callingRoutine + " in uy[" << ix << "," << iy << "," << iz << "]" << std::endl;
					exit(-1);
				}
				if (std::abs(Fy[getFaceIndexY(ix, iy, iz)])>1e10) {
					std::cerr << "error in routine " + callingRoutine + " in Fy[" << ix << "," << iy << "," << iz << "]" << std::endl;
					exit(-1);
				}
			}

	for (int ix = 0; ix<numberOfCellsPerAxisX + 2; ix++)
		for (int iy = 0; iy<numberOfCellsPerAxisY + 2; iy++)
			for (int iz = 0; iz<numberOfCellsPerAxisZ + 3; iz++) {
				if (std::abs(uz[getFaceIndexZ(ix, iy, iz)])>1e10) {
					std::cerr << "error in in routine " + callingRoutine + " uz[" << ix << "," << iy << "," << iz << "]: " << uz[getFaceIndexZ(ix, iy, iz)] << std::endl;
					exit(-1);
				}
				if (std::abs(Fz[getFaceIndexZ(ix, iy, iz)])>1e10) {
					std::cerr << "error in in routine " + callingRoutine + " Fz[" << ix << "," << iy << "," << iz << "]" << std::endl;
					exit(-1);
				}
			}
#endif
}

vtkSmartPointer<vtkRenderer> makeGridImageData(vtkSmartPointer<vtkImageData> imageData) {

	// Number of points is 1 more than number of cells
	imageData->SetDimensions(numberOfCellsPerAxisX+1, numberOfCellsPerAxisY+1, numberOfCellsPerAxisZ+1);

	vtkSmartPointer<vtkDoubleArray> u = vtkSmartPointer<vtkDoubleArray>::New();
	u->SetName("velocity");
	u->SetNumberOfComponents(3);
	u->SetNumberOfTuples(imageData->GetNumberOfPoints());
	for (int iz = 1; iz<numberOfCellsPerAxisZ + 2; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 2; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 2; ix++) {
				u->SetTuple3(getVertexIndex(ix - 1, iy - 1, iz - 1),
					0.25 * (ux[getFaceIndexX(ix, iy - 1, iz - 1)] + ux[getFaceIndexX(ix, iy - 0, iz - 1)] + ux[getFaceIndexX(ix, iy - 1, iz - 0)] + ux[getFaceIndexX(ix, iy, iz)]),
					0.25 * (uy[getFaceIndexY(ix - 1, iy, iz - 1)] + uy[getFaceIndexY(ix - 0, iy, iz - 1)] + uy[getFaceIndexY(ix - 1, iy, iz - 0)] + uy[getFaceIndexY(ix, iy, iz)]),
					0.25 * (uz[getFaceIndexZ(ix - 1, iy - 1, iz)] + uz[getFaceIndexZ(ix - 0, iy - 1, iz)] + uz[getFaceIndexZ(ix - 1, iy - 0, iz)] + uz[getFaceIndexZ(ix, iy, iz)]));
			}
		}
	}

	imageData->GetPointData()->SetVectors(u);

	vtkSmartPointer<vtkDoubleArray> pressures = vtkSmartPointer<vtkDoubleArray>::New();
	pressures->SetName("Pressure");
	pressures->SetNumberOfValues(imageData->GetNumberOfCells());

	maxPressure = 0;
	minPressure = 0;

	for (int iz = 1; iz<numberOfCellsPerAxisZ + 1; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 1; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 1; ix++) {
				double index = numberOfCellsPerAxisX*numberOfCellsPerAxisY*(iz - 1) + numberOfCellsPerAxisX*(iy - 1) + (ix - 1);
				double iPressure = p[getCellIndex(ix, iy, iz)];
				if (iPressure > maxPressure) {
					maxPressure = iPressure;
				}
				if (iPressure < minPressure) {
					minPressure = iPressure;
				}

				pressures->SetValue(index, p[getCellIndex(ix, iy, iz)]);
			}
		}
	}

	cout << minPressure << endl;
	cout << maxPressure << endl;

	imageData->GetCellData()->SetScalars(pressures);

	// Create the color map
	colorLookupTable = vtkSmartPointer<vtkLookupTable>::New();
	//colorLookupTable->SetNumberOfTableValues(64);
	//colorLookupTable->SetHueRange(0.0, 0.667);
	//colorLookupTable->SetTableRange(minPressure, maxPressure);
	//colorLookupTable->Build();

	//// Generate the colors for each point based on the color map
	//vtkSmartPointer<vtkUnsignedCharArray> colors =
	//	vtkSmartPointer<vtkUnsignedCharArray>::New();
	//colors->SetNumberOfComponents(3);
	//colors->SetName("Colors");

	//for (int i = 0; i < imageData->GetNumberOfCells(); i++) {
	//	
	//	double pressureColour[3];
	//	colorLookupTable->GetColor(imageData->GetCellData()->get, pressureColour);

	//	double pressure;
	//	imageData->GetCell(i)->;

	//}

	vtkSmartPointer<vtkAssignAttribute> scalars =
		vtkSmartPointer<vtkAssignAttribute>::New();
	scalars->SetInputData(imageData);
	scalars->Assign("Pressure", vtkDataSetAttributes::SCALARS,
		vtkAssignAttribute::CELL_DATA);

	vtkSmartPointer<vtkAssignAttribute> vectors =
		vtkSmartPointer<vtkAssignAttribute>::New();
	vectors->SetInputData(imageData);
	vectors->Assign("velocity", vtkDataSetAttributes::VECTORS,
		vtkAssignAttribute::POINT_DATA);

	vtkSmartPointer<vtkArrowSource> arrowSource = vtkSmartPointer<vtkArrowSource>::New();

	vtkSmartPointer<vtkGlyph3D> glyphs =
		vtkSmartPointer<vtkGlyph3D>::New();
	glyphs->SetInputConnection(0, vectors->GetOutputPort());
	glyphs->SetInputConnection(1, arrowSource->GetOutputPort());
	glyphs->ScalingOn();
	//glyphs->SetScaleModeToScaleByVector();
	glyphs->SetScaleFactor(0.5);
	glyphs->OrientOn();
	glyphs->ClampingOff();
	glyphs->SetVectorModeToUseVector();
	glyphs->SetIndexModeToOff();
	glyphs->Update();

	// Visualize vectors
	vtkSmartPointer<vtkPolyDataMapper> mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(glyphs->GetOutputPort());
	vtkSmartPointer<vtkActor> actor =
		vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	// Visualise scalars
	pressureMapper =
		vtkSmartPointer<vtkDataSetMapper>::New();
	pressureMapper->SetInputData(imageData);
	//pressureMapper->SetScalarVisibility(true);
	//pressureMapper->SetScalarModeToUseCellData();
	pressureMapper->SetLookupTable(colorLookupTable);
	//pressureMapper->SetColorModeToMapScalars();
	pressureMapper->SetScalarRange(minPressure, maxPressure);
	//pressureMapper->CreateDefaultLookupTable();
	vtkSmartPointer<vtkActor> pressureActor =
		vtkSmartPointer<vtkActor>::New();
	pressureActor->SetMapper(pressureMapper);

	// Visualize
	vtkSmartPointer<vtkRenderer> renderer =
		vtkSmartPointer<vtkRenderer>::New();

	renderer->AddActor(actor);
	renderer->AddActor(pressureActor);
	renderer->SetBackground(.2, .3, .4);

	return renderer;
	
}

void updateVisualisation(vtkSmartPointer<vtkImageData> imageData) {
	vtkSmartPointer<vtkDoubleArray> u = vtkSmartPointer<vtkDoubleArray>::New();
	u->SetName("velocity");
	u->SetNumberOfComponents(3);
	u->SetNumberOfTuples(imageData->GetNumberOfPoints());
	for (int iz = 1; iz<numberOfCellsPerAxisZ + 2; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 2; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 2; ix++) {
				u->SetTuple3(getVertexIndex(ix - 1, iy - 1, iz - 1),
					0.25 * (ux[getFaceIndexX(ix, iy - 1, iz - 1)] + ux[getFaceIndexX(ix, iy - 0, iz - 1)] + ux[getFaceIndexX(ix, iy - 1, iz - 0)] + ux[getFaceIndexX(ix, iy, iz)]),
					0.25 * (uy[getFaceIndexY(ix - 1, iy, iz - 1)] + uy[getFaceIndexY(ix - 0, iy, iz - 1)] + uy[getFaceIndexY(ix - 1, iy, iz - 0)] + uy[getFaceIndexY(ix, iy, iz)]),
					0.25 * (uz[getFaceIndexZ(ix - 1, iy - 1, iz)] + uz[getFaceIndexZ(ix - 0, iy - 1, iz)] + uz[getFaceIndexZ(ix - 1, iy - 0, iz)] + uz[getFaceIndexZ(ix, iy, iz)]));
			}
		}
	}

	imageData->GetPointData()->SetVectors(u);

	vtkSmartPointer<vtkDoubleArray> pressures = vtkSmartPointer<vtkDoubleArray>::New();
	pressures->SetName("Pressure");
	pressures->SetNumberOfValues(imageData->GetNumberOfCells());

	maxPressure = 0;
	minPressure = 0;

	for (int iz = 1; iz<numberOfCellsPerAxisZ + 1; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 1; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 1; ix++) {
				double index = numberOfCellsPerAxisX*numberOfCellsPerAxisY*(iz - 1) + numberOfCellsPerAxisX*(iy - 1) + (ix - 1);
				double iPressure = p[getCellIndex(ix, iy, iz)];
				if (iPressure > maxPressure) {
					maxPressure = iPressure;
				}
				if (iPressure < minPressure) {
					minPressure = iPressure;
				}

				pressures->SetValue(index, p[getCellIndex(ix, iy, iz)]);
			}
		}
	}

	cout << minPressure << endl;
	cout << maxPressure << endl;

	imageData->GetCellData()->SetScalars(pressures);
	colorLookupTable->SetTableRange(minPressure, maxPressure);
	colorLookupTable->Build();
	pressureMapper->SetScalarRange(minPressure, maxPressure);

}

/**
* Plot a vtk file. This function probably never has to be changed when you do
* your assessment.
*/
void plotVTKFile() {
	static int vtkFileCounter = 0;

	std::ostringstream  outputFileName;
	outputFileName << "output-" << vtkFileCounter << ".vtk";
	std::ofstream out;
	out.open(outputFileName.str().c_str());

	std::cout << "\t write " << outputFileName.str();

	out << "# vtk DataFile Version 2.0" << std::endl
		<< "Tobias Weinzierl: CPU, Manycore and Cluster Computing" << std::endl
		<< "ASCII" << std::endl << std::endl;

	out << "DATASET STRUCTURED_POINTS" << std::endl
		<< "DIMENSIONS  "
		<< numberOfCellsPerAxisX + 1 << " "
		<< numberOfCellsPerAxisY + 1 << " "
		<< numberOfCellsPerAxisZ + 1
		<< std::endl << std::endl;
	out << "ORIGIN 0 0 0 " << std::endl << std::endl;
	out << "SPACING "
		<< getH() << " "
		<< getH() << " "
		<< getH() << " "
		<< std::endl << std::endl;

	const int numberOfVertices = (numberOfCellsPerAxisX + 1) * (numberOfCellsPerAxisY + 1) * (numberOfCellsPerAxisZ + 1);
	out << "POINT_DATA " << numberOfVertices << std::endl << std::endl;


	out << "VECTORS velocity float" << std::endl;
	for (int iz = 1; iz<numberOfCellsPerAxisZ + 2; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 2; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 2; ix++) {
				out << 0.25 * (ux[getFaceIndexX(ix, iy - 1, iz - 1)] + ux[getFaceIndexX(ix, iy - 0, iz - 1)] + ux[getFaceIndexX(ix, iy - 1, iz - 0)] + ux[getFaceIndexX(ix, iy, iz)]) << " ";
				out << 0.25 * (uy[getFaceIndexY(ix - 1, iy, iz - 1)] + uy[getFaceIndexY(ix - 0, iy, iz - 1)] + uy[getFaceIndexY(ix - 1, iy, iz - 0)] + uy[getFaceIndexY(ix, iy, iz)]) << " ";
				out << 0.25 * (uz[getFaceIndexZ(ix - 1, iy - 1, iz)] + uz[getFaceIndexZ(ix - 0, iy - 1, iz)] + uz[getFaceIndexZ(ix - 1, iy - 0, iz)] + uz[getFaceIndexZ(ix, iy, iz)]) << std::endl;
			}
		}
	}
	out << std::endl << std::endl;

	//
	// For debugging, it sometimes pays off to see F. Usually not required - notably not for this year's
	// assignments
	//
#ifdef CheckVariableValues
	out << "VECTORS F float" << std::endl;
	for (int iz = 1; iz<numberOfCellsPerAxisZ + 2; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 2; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 2; ix++) {
				out << 0.25 * (Fx[getFaceIndexX(ix, iy - 1, iz - 1)] + Fx[getFaceIndexX(ix, iy - 0, iz - 1)] + Fx[getFaceIndexX(ix, iy - 1, iz - 0)] + Fx[getFaceIndexX(ix, iy, iz)]) << " ";
				out << 0.25 * (Fy[getFaceIndexY(ix - 1, iy, iz - 1)] + Fy[getFaceIndexY(ix - 0, iy, iz - 1)] + Fy[getFaceIndexY(ix - 1, iy, iz - 0)] + Fy[getFaceIndexY(ix, iy, iz)]) << " ";
				out << 0.25 * (Fz[getFaceIndexZ(ix - 1, iy - 1, iz)] + Fz[getFaceIndexZ(ix - 0, iy - 1, iz)] + Fz[getFaceIndexZ(ix - 1, iy - 0, iz)] + Fz[getFaceIndexZ(ix, iy, iz)]) << std::endl;
			}
		}
	}
	out << std::endl << std::endl;
#endif

	out << std::endl << std::endl;

	const int numberOfCells = numberOfCellsPerAxisX * numberOfCellsPerAxisY * numberOfCellsPerAxisZ;
	out << "CELL_DATA " << numberOfCells << std::endl << std::endl;

	out << "SCALARS pressure float 1" << std::endl;
	out << "LOOKUP_TABLE default" << std::endl;

	for (int iz = 1; iz<numberOfCellsPerAxisZ + 1; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 1; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 1; ix++) {
				out << p[getCellIndex(ix, iy, iz)] << std::endl;
			}
		}
	}

	//
	// For debugging, it sometimes pays off to see the rhs of the pressure
	// Poisson equation. Usually not required - notably not for this year's
	// assignments
	//
#ifdef CheckVariableValues
	out << "SCALARS rhs float 1" << std::endl;
	out << "LOOKUP_TABLE default" << std::endl;

	for (int iz = 1; iz<numberOfCellsPerAxisZ + 1; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 1; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 1; ix++) {
				out << rhs[getCellIndex(ix, iy, iz)] << std::endl;
			}
		}
	}
#endif

	out.close();

	vtkFileCounter++;
}


/**
* Computes a helper velocity. See book of Griebel for details.
*/
void computeF() {
	const double alpha = timeStepSize / getH();

	for (int iz = 1; iz<numberOfCellsPerAxisZ + 2 - 1; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 2 - 1; iy++) {
			for (int ix = 2; ix<numberOfCellsPerAxisX + 3 - 2; ix++) {
				const double diffusiveTerm =
					+(-1.0 * ux[getFaceIndexX(ix - 1, iy, iz)] + 2.0 * ux[getFaceIndexX(ix, iy, iz)] - 1.0 * ux[getFaceIndexX(ix + 1, iy, iz)])
					+ (-1.0 * ux[getFaceIndexX(ix, iy - 1, iz)] + 2.0 * ux[getFaceIndexX(ix, iy, iz)] - 1.0 * ux[getFaceIndexX(ix, iy + 1, iz)])
					+ (-1.0 * ux[getFaceIndexX(ix, iy, iz - 1)] + 2.0 * ux[getFaceIndexX(ix, iy, iz)] - 1.0 * ux[getFaceIndexX(ix, iy, iz + 1)]);

				const double convectiveTerm =
					+((ux[getFaceIndexX(ix, iy, iz)] + ux[getFaceIndexX(ix + 1, iy, iz)])*(ux[getFaceIndexX(ix, iy, iz)] + ux[getFaceIndexX(ix + 1, iy, iz)]) - (ux[getFaceIndexX(ix - 1, iy, iz)] + ux[getFaceIndexX(ix, iy, iz)])    *(ux[getFaceIndexX(ix - 1, iy, iz)] + ux[getFaceIndexX(ix, iy, iz)]))
					+ ((uy[getFaceIndexY(ix, iy, iz)] + uy[getFaceIndexY(ix + 1, iy, iz)])*(ux[getFaceIndexX(ix, iy, iz)] + ux[getFaceIndexX(ix, iy + 1, iz)]) - (uy[getFaceIndexY(ix, iy - 1, iz)] + uy[getFaceIndexY(ix + 1, iy - 1, iz)])*(ux[getFaceIndexX(ix, iy - 1, iz)] + ux[getFaceIndexX(ix, iy, iz)]))
					+ ((uz[getFaceIndexZ(ix, iy, iz)] + uz[getFaceIndexZ(ix + 1, iy, iz)])*(ux[getFaceIndexX(ix, iy, iz)] + ux[getFaceIndexX(ix, iy, iz + 1)]) - (uz[getFaceIndexZ(ix, iy, iz - 1)] + uz[getFaceIndexZ(ix + 1, iy, iz - 1)])*(ux[getFaceIndexX(ix, iy, iz - 1)] + ux[getFaceIndexX(ix, iy, iz)]))
					+ alpha * (std::abs(ux[getFaceIndexX(ix, iy, iz)] + ux[getFaceIndexX(ix + 1, iy, iz)])*(ux[getFaceIndexX(ix, iy, iz)] - ux[getFaceIndexX(ix + 1, iy, iz)]) - std::abs(ux[getFaceIndexX(ix - 1, iy, iz)] + ux[getFaceIndexX(ix, iy, iz)])    *(ux[getFaceIndexX(ix - 1, iy, iz)] - ux[getFaceIndexX(ix, iy, iz)]))
					+ alpha * (std::abs(uy[getFaceIndexY(ix, iy, iz)] + uy[getFaceIndexY(ix + 1, iy, iz)])*(ux[getFaceIndexX(ix, iy, iz)] - ux[getFaceIndexX(ix, iy + 1, iz)]) - std::abs(uy[getFaceIndexY(ix, iy - 1, iz)] + uy[getFaceIndexY(ix + 1, iy - 1, iz)])*(ux[getFaceIndexX(ix, iy - 1, iz)] - ux[getFaceIndexX(ix, iy, iz)]))
					+ alpha * (std::abs(uz[getFaceIndexZ(ix, iy, iz)] + uz[getFaceIndexZ(ix + 1, iy, iz)])*(ux[getFaceIndexX(ix, iy, iz)] - ux[getFaceIndexX(ix, iy, iz + 1)]) - std::abs(uz[getFaceIndexZ(ix, iy, iz - 1)] + uz[getFaceIndexZ(ix + 1, iy, iz - 1)])*(ux[getFaceIndexX(ix, iy, iz - 1)] - ux[getFaceIndexX(ix, iy, iz)]))
					;

				Fx[getFaceIndexX(ix, iy, iz)] =
					ux[getFaceIndexX(ix, iy, iz)]
					- timeStepSize / ReynoldsNumber * 1.0 / getH() / getH() * diffusiveTerm
					- timeStepSize * 1.0 / getH() / 4.0 * convectiveTerm;
			}
		}
	}

	for (int iz = 1; iz<numberOfCellsPerAxisZ + 2 - 1; iz++) {
		for (int iy = 2; iy<numberOfCellsPerAxisY + 3 - 2; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 2 - 1; ix++) {
				const double diffusiveTerm =
					+(-1.0 * uy[getFaceIndexY(ix - 1, iy, iz)] + 2.0 * uy[getFaceIndexY(ix, iy, iz)] - 1.0 * uy[getFaceIndexY(ix + 1, iy, iz)])
					+ (-1.0 * uy[getFaceIndexY(ix, iy - 1, iz)] + 2.0 * uy[getFaceIndexY(ix, iy, iz)] - 1.0 * uy[getFaceIndexY(ix, iy + 1, iz)])
					+ (-1.0 * uy[getFaceIndexY(ix, iy, iz - 1)] + 2.0 * uy[getFaceIndexY(ix, iy, iz)] - 1.0 * uy[getFaceIndexY(ix, iy, iz + 1)])
					;

				const double convectiveTerm =
					+((ux[getFaceIndexX(ix, iy, iz)] + ux[getFaceIndexX(ix, iy + 1, iz)])*(uy[getFaceIndexY(ix, iy, iz)] + uy[getFaceIndexY(ix + 1, iy, iz)]) - (ux[getFaceIndexX(ix - 1, iy, iz)] + ux[getFaceIndexX(ix - 1, iy + 1, iz)]) *(uy[getFaceIndexY(ix - 1, iy, iz)] + uy[getFaceIndexY(ix, iy, iz)]))
					+ ((uy[getFaceIndexY(ix, iy, iz)] + uy[getFaceIndexY(ix, iy + 1, iz)])*(uy[getFaceIndexY(ix, iy, iz)] + uy[getFaceIndexY(ix, iy + 1, iz)]) - (uy[getFaceIndexY(ix, iy - 1, iz)] + uy[getFaceIndexY(ix, iy, iz)])     *(uy[getFaceIndexY(ix, iy - 1, iz)] + uy[getFaceIndexY(ix, iy, iz)]))
					+ ((uz[getFaceIndexZ(ix, iy, iz)] + uz[getFaceIndexZ(ix, iy + 1, iz)])*(uy[getFaceIndexY(ix, iy, iz)] + uy[getFaceIndexY(ix, iy, iz + 1)]) - (uz[getFaceIndexZ(ix, iy, iz - 1)] + uz[getFaceIndexZ(ix, iy + 1, iz - 1)]) *(uy[getFaceIndexY(ix, iy, iz - 1)] + uy[getFaceIndexY(ix, iy, iz)]))
					+ alpha * (std::abs(ux[getFaceIndexX(ix, iy, iz)] + ux[getFaceIndexX(ix, iy + 1, iz)])*(uy[getFaceIndexY(ix, iy, iz)] - uy[getFaceIndexY(ix + 1, iy, iz)]) - std::abs(ux[getFaceIndexX(ix - 1, iy, iz)] + ux[getFaceIndexX(ix - 1, iy + 1, iz)]) *(uy[getFaceIndexY(ix - 1, iy, iz)] - uy[getFaceIndexY(ix, iy, iz)]))
					+ alpha * (std::abs(uy[getFaceIndexY(ix, iy, iz)] + uy[getFaceIndexY(ix, iy + 1, iz)])*(uy[getFaceIndexY(ix, iy, iz)] - uy[getFaceIndexY(ix, iy + 1, iz)]) - std::abs(uy[getFaceIndexY(ix, iy - 1, iz)] + uy[getFaceIndexY(ix, iy, iz)])     *(uy[getFaceIndexY(ix, iy - 1, iz)] - uy[getFaceIndexY(ix, iy, iz)]))
					+ alpha * (std::abs(uz[getFaceIndexZ(ix, iy, iz)] + uz[getFaceIndexZ(ix, iy + 1, iz)])*(uy[getFaceIndexY(ix, iy, iz)] - uy[getFaceIndexY(ix, iy, iz + 1)]) - std::abs(uz[getFaceIndexZ(ix, iy, iz - 1)] + uz[getFaceIndexZ(ix, iy + 1, iz - 1)]) *(uy[getFaceIndexY(ix, iy, iz - 1)] - uy[getFaceIndexY(ix, iy, iz)]))
					;

				Fy[getFaceIndexY(ix, iy, iz)] =
					uy[getFaceIndexY(ix, iy, iz)]
					- timeStepSize / ReynoldsNumber * 1.0 / getH() / getH() * diffusiveTerm
					- timeStepSize * 1.0 / getH() / 4.0 * convectiveTerm;
			}
		}
	}

	for (int iz = 2; iz<numberOfCellsPerAxisZ + 3 - 2; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 2 - 1; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 2 - 1; ix++) {
				const double diffusiveTerm =
					+(-1.0 * uz[getFaceIndexZ(ix - 1, iy, iz)] + 2.0 * uz[getFaceIndexZ(ix, iy, iz)] - 1.0 * uz[getFaceIndexZ(ix + 1, iy, iz)])
					+ (-1.0 * uz[getFaceIndexZ(ix, iy - 1, iz)] + 2.0 * uz[getFaceIndexZ(ix, iy, iz)] - 1.0 * uz[getFaceIndexZ(ix, iy + 1, iz)])
					+ (-1.0 * uz[getFaceIndexZ(ix, iy, iz - 1)] + 2.0 * uz[getFaceIndexZ(ix, iy, iz)] - 1.0 * uz[getFaceIndexZ(ix, iy, iz + 1)])
					;

				const double convectiveTerm =
					+((ux[getFaceIndexX(ix, iy, iz)] + ux[getFaceIndexX(ix, iy, iz + 1)])*(uz[getFaceIndexZ(ix, iy, iz)] + uz[getFaceIndexZ(ix + 1, iy, iz)]) - (ux[getFaceIndexX(ix - 1, iy, iz)] + ux[getFaceIndexX(ix - 1, iy, iz + 1)]) *(uz[getFaceIndexZ(ix - 1, iy, iz)] + uz[getFaceIndexZ(ix, iy, iz)]))
					+ ((uy[getFaceIndexY(ix, iy, iz)] + uy[getFaceIndexY(ix, iy, iz + 1)])*(uz[getFaceIndexZ(ix, iy, iz)] + uz[getFaceIndexZ(ix, iy + 1, iz)]) - (uy[getFaceIndexY(ix, iy - 1, iz)] + uy[getFaceIndexY(ix, iy - 1, iz + 1)]) *(uz[getFaceIndexZ(ix, iy - 1, iz)] + uz[getFaceIndexZ(ix, iy, iz)]))
					+ ((uz[getFaceIndexZ(ix, iy, iz)] + uz[getFaceIndexZ(ix, iy, iz + 1)])*(uz[getFaceIndexZ(ix, iy, iz)] + uz[getFaceIndexZ(ix, iy, iz + 1)]) - (uz[getFaceIndexZ(ix, iy, iz - 1)] + uz[getFaceIndexZ(ix, iy, iz)])     *(uz[getFaceIndexZ(ix, iy, iz - 1)] + uz[getFaceIndexZ(ix, iy, iz)]))
					+ alpha * (std::abs(ux[getFaceIndexX(ix, iy, iz)] + ux[getFaceIndexX(ix, iy, iz + 1)])*(uz[getFaceIndexZ(ix, iy, iz)] - uz[getFaceIndexZ(ix + 1, iy, iz)]) - std::abs(ux[getFaceIndexX(ix - 1, iy, iz)] + ux[getFaceIndexX(ix - 1, iy, iz + 1)]) *(uz[getFaceIndexZ(ix - 1, iy, iz)] - uz[getFaceIndexZ(ix, iy, iz)]))
					+ alpha * (std::abs(uy[getFaceIndexY(ix, iy, iz)] + uy[getFaceIndexY(ix, iy, iz + 1)])*(uz[getFaceIndexZ(ix, iy, iz)] - uz[getFaceIndexZ(ix, iy + 1, iz)]) - std::abs(uy[getFaceIndexY(ix, iy - 1, iz)] + uy[getFaceIndexY(ix, iy - 1, iz + 1)]) *(uz[getFaceIndexZ(ix, iy - 1, iz)] - uz[getFaceIndexZ(ix, iy, iz)]))
					+ alpha * (std::abs(uz[getFaceIndexZ(ix, iy, iz)] + uz[getFaceIndexZ(ix, iy, iz + 1)])*(uz[getFaceIndexZ(ix, iy, iz)] - uz[getFaceIndexZ(ix, iy, iz + 1)]) - std::abs(uz[getFaceIndexZ(ix, iy, iz - 1)] + uz[getFaceIndexZ(ix, iy, iz)])     *(uz[getFaceIndexZ(ix, iy, iz - 1)] - uz[getFaceIndexZ(ix, iy, iz)]))
					;

				Fz[getFaceIndexZ(ix, iy, iz)] =
					uz[getFaceIndexZ(ix, iy, iz)]
					- timeStepSize / ReynoldsNumber * 1.0 / getH() / getH() * diffusiveTerm
					- timeStepSize * 1.0 / getH() / 4.0 * convectiveTerm;
			}
		}
	}

	validateThatEntriesAreBounded("computeF");
}


/**
* Compute the right-hand side. This basically means how much a flow would
* violate the incompressibility if there were no pressure.
*/
void computeRhs() {
	for (int iz = 1; iz<numberOfCellsPerAxisZ + 2 - 1; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 2 - 1; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 2 - 1; ix++) {
				rhs[getCellIndex(ix, iy, iz)] = 1.0 / timeStepSize / getH()*
					(
						Fx[getFaceIndexX(ix + 1, iy, iz)] - Fx[getFaceIndexX(ix, iy, iz)] +
						Fy[getFaceIndexY(ix, iy + 1, iz)] - Fy[getFaceIndexY(ix, iy, iz)] +
						Fz[getFaceIndexZ(ix, iy, iz + 1)] - Fz[getFaceIndexZ(ix, iy, iz)]
						);
			}
		}
	}
}


/**
* Set boundary conditions for pressure. The values of the pressure at the
* domain boundary might depend on the pressure itself. So if we update it
* in the algorithm, we afterwards have to reset the boundary conditions
* again.
*/
void setPressureBoundaryConditions() {
	int ix, iy, iz;

	// Neumann Boundary conditions for p
	for (iy = 0; iy<numberOfCellsPerAxisY + 2; iy++) {
		for (iz = 0; iz<numberOfCellsPerAxisZ + 2; iz++) {
			ix = 0;
			p[getCellIndex(ix, iy, iz)] = p[getCellIndex(ix + 1, iy, iz)];
			ix = numberOfCellsPerAxisX + 1;
			p[getCellIndex(ix, iy, iz)] = p[getCellIndex(ix - 1, iy, iz)];
		}
	}
	for (ix = 0; ix<numberOfCellsPerAxisX + 2; ix++) {
		for (iz = 0; iz<numberOfCellsPerAxisZ + 2; iz++) {
			iy = 0;
			p[getCellIndex(ix, iy, iz)] = p[getCellIndex(ix, iy + 1, iz)];
			iy = numberOfCellsPerAxisY + 1;
			p[getCellIndex(ix, iy, iz)] = p[getCellIndex(ix, iy - 1, iz)];
		}
	}
	for (ix = 0; ix<numberOfCellsPerAxisX + 2; ix++) {
		for (iy = 0; iy<numberOfCellsPerAxisY + 2; iy++) {
			iz = 0;
			p[getCellIndex(ix, iy, iz)] = p[getCellIndex(ix, iy, iz + 1)];
			iz = numberOfCellsPerAxisZ + 1;
			p[getCellIndex(ix, iy, iz)] = p[getCellIndex(ix, iy, iz - 1)];
		}
	}

	// Normalise pressure at rhs to zero
	for (iy = 1; iy<numberOfCellsPerAxisY + 2 - 1; iy++) {
		for (iz = 1; iz<numberOfCellsPerAxisZ + 2 - 1; iz++) {
			p[getCellIndex(numberOfCellsPerAxisX + 1, iy, iz)] = 0.0;
		}
	}
}


/**
* Determine the new pressure. The operation depends on properly set pressure
* boundary conditions. See setPressureBoundaryConditions().
*
* @return Number of iterations required or max number plus one if we had to
*         stop iterating as the solver diverged.
*/
int computeP() {
	double       globalResidual = 1.0;
	double       firstResidual = 1.0;
	double       previousGlobalResidual = 2.0;
	int          iterations = 0;

	while (
		(
			std::abs(globalResidual - previousGlobalResidual)>PPESolverThreshold
			&&
			iterations<MaxComputePIterations
			&&
			std::abs(globalResidual)>PPESolverThreshold
			&&
			(globalResidual / firstResidual>PPESolverThreshold)
			)
		||
		(iterations % 2 == 1) // we have alternating omega, so we allow only even iteration counts
		) {
		const double omega = iterations % 2 == 0 ? 1.2 : 0.8;
		setPressureBoundaryConditions();

		previousGlobalResidual = globalResidual;
		globalResidual = 0.0;
		for (int iz = 1; iz<numberOfCellsPerAxisZ + 1; iz++) {
			for (int iy = 1; iy<numberOfCellsPerAxisY + 1; iy++) {
				for (int ix = 1; ix<numberOfCellsPerAxisX + 1; ix++) {
					double residual = rhs[getCellIndex(ix, iy, iz)] +
						1.0 / getH() / getH()*
						(
							-1.0 * p[getCellIndex(ix - 1, iy, iz)]
							- 1.0 * p[getCellIndex(ix + 1, iy, iz)]
							- 1.0 * p[getCellIndex(ix, iy - 1, iz)]
							- 1.0 * p[getCellIndex(ix, iy + 1, iz)]
							- 1.0 * p[getCellIndex(ix, iy, iz - 1)]
							- 1.0 * p[getCellIndex(ix, iy, iz + 1)]
							+ 6.0 * p[getCellIndex(ix, iy, iz)]
							);
					globalResidual += residual * residual;
					p[getCellIndex(ix, iy, iz)] += -omega * residual / 6.0 * getH() * getH();
				}
			}
		}
		globalResidual = std::sqrt(globalResidual);
		firstResidual = firstResidual == 0 ? globalResidual : firstResidual;
		iterations++;
	}

	std::cout << "iterations n=" << iterations
		<< ", |res(n)|_2=" << globalResidual
		<< ", |res(n-1)|_2=" << previousGlobalResidual
		<< ", |res(n-1)|_2-|res(n)|_2=" << (previousGlobalResidual - globalResidual);

	return iterations;
}


/**
* @todo Your job if you attend the Scientific Computing submodule. Otherwise empty.
*/
void updateInk() {
}



/**
* Once we have F and a valid pressure p, we may update the velocities.
*/
void setNewVelocities() {
	for (int iz = 1; iz<numberOfCellsPerAxisZ + 2 - 1; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 2 - 1; iy++) {
			for (int ix = 2; ix<numberOfCellsPerAxisX + 3 - 2; ix++) {
				ux[getFaceIndexX(ix, iy, iz)] = Fx[getFaceIndexX(ix, iy, iz)] - timeStepSize / getH() * (p[getCellIndex(ix, iy, iz)] - p[getCellIndex(ix - 1, iy, iz)]);
			}
		}
	}

	for (int iz = 1; iz<numberOfCellsPerAxisZ + 2 - 1; iz++) {
		for (int iy = 2; iy<numberOfCellsPerAxisY + 3 - 2; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 2 - 1; ix++) {
				uy[getFaceIndexY(ix, iy, iz)] = Fy[getFaceIndexY(ix, iy, iz)] - timeStepSize / getH() * (p[getCellIndex(ix, iy, iz)] - p[getCellIndex(ix, iy - 1, iz)]);
			}
		}
	}

	for (int iz = 2; iz<numberOfCellsPerAxisZ + 3 - 2; iz++) {
		for (int iy = 1; iy<numberOfCellsPerAxisY + 2 - 1; iy++) {
			for (int ix = 1; ix<numberOfCellsPerAxisX + 2 - 1; ix++) {
				uz[getFaceIndexZ(ix, iy, iz)] = Fz[getFaceIndexZ(ix, iy, iz)] - timeStepSize / getH() * (p[getCellIndex(ix, iy, iz)] - p[getCellIndex(ix, iy, iz - 1)]);
			}
		}
	}
}


/**
* Setup our scenario, i.e. initialise all the big arrays and set the
* right boundary conditions. This is something you might want to change in
* part three of the assessment.
*/
void setupScenario() {
	const int numberOfCells = (numberOfCellsPerAxisX + 2) * (numberOfCellsPerAxisY + 2) * (numberOfCellsPerAxisZ + 2);

	const int numberOfFacesX = (numberOfCellsPerAxisX + 3) * (numberOfCellsPerAxisY + 2) * (numberOfCellsPerAxisZ + 2);
	const int numberOfFacesY = (numberOfCellsPerAxisX + 2) * (numberOfCellsPerAxisY + 3) * (numberOfCellsPerAxisZ + 2);
	const int numberOfFacesZ = (numberOfCellsPerAxisX + 2) * (numberOfCellsPerAxisY + 2) * (numberOfCellsPerAxisZ + 3);

	ux = 0;
	uy = 0;
	uz = 0;
	Fx = 0;
	Fy = 0;
	Fz = 0;
	p = 0;
	rhs = 0;

	ux = new (std::nothrow) double[numberOfFacesX];
	uy = new (std::nothrow) double[numberOfFacesY];
	uz = new (std::nothrow) double[numberOfFacesZ];
	Fx = new (std::nothrow) double[numberOfFacesX];
	Fy = new (std::nothrow) double[numberOfFacesY];
	Fz = new (std::nothrow) double[numberOfFacesZ];

	p = new (std::nothrow) double[numberOfCells];
	rhs = new (std::nothrow) double[numberOfCells];

	if (
		ux == 0 ||
		uy == 0 ||
		uz == 0 ||
		Fx == 0 ||
		Fy == 0 ||
		Fz == 0 ||
		p == 0 ||
		rhs == 0
		) {
		std::cerr << "could not allocate memory. Perhaps not enough memory free?" << std::endl;
		exit(-1);
	}


	for (int i = 0; i<numberOfCells; i++) {
		p[i] = 0.0;
	}

	for (int i = 0; i<numberOfFacesX; i++) {
		ux[i] = 0;
		Fx[i] = 0;
	}
	for (int i = 0; i<numberOfFacesY; i++) {
		uy[i] = 0;
		Fy[i] = 0;
	}
	for (int i = 0; i<numberOfFacesZ; i++) {
		uz[i] = 0;
		Fz[i] = 0;
	}


	validateThatEntriesAreBounded("setupScenario()");
}


/**
* Clean up the system
*/
void freeDataStructures() {
	delete[] p;
	delete[] rhs;
	delete[] ux;
	delete[] uy;
	delete[] uz;
	delete[] Fx;
	delete[] Fy;
	delete[] Fz;

	ux = 0;
	uy = 0;
	uz = 0;
	Fx = 0;
	Fy = 0;
	Fz = 0;
	p = 0;
	rhs = 0;
}


/**
* - Handle all the velocities at the domain boundaries. We either
*   realise no-slip or free-slip.
* - Fix all the F values. Along the boundary, the F values equal the
*   velocity values.
*
*/
void setVelocityBoundaryConditions(double time) {
	int ix, iy, iz;

	validateThatEntriesAreBounded("setVelocityBoundaryConditions(double)[in]");

	const bool UseNoSlip = true;
	// We ensure that no fluid leaves the domain. For this, we set the velocities
	// along the boundary to zero. Furthermore,  we ensure that the tangential
	// components of all velocities are zero. For this, we set the ghost/virtual
	// velocities to minus the original one. If we interpolate linearly in-between,
	// we obtain zero tangential speed.
	for (iy = 0; iy<numberOfCellsPerAxisY + 2; iy++) {
		for (iz = 0; iz<numberOfCellsPerAxisZ + 2; iz++) {
			ix = 0;
			ux[getFaceIndexX(ix, iy, iz)] = 0.0;
			ix = 1;
			ux[getFaceIndexX(ix, iy, iz)] = 0.0;

			ix = 0;
			uy[getFaceIndexY(ix, iy, iz)] = (UseNoSlip ? -1.0 : 1.0) * uy[getFaceIndexY(ix + 1, iy, iz)];
			uz[getFaceIndexZ(ix, iy, iz)] = (UseNoSlip ? -1.0 : 1.0) * uz[getFaceIndexZ(ix + 1, iy, iz)];

			ix = numberOfCellsPerAxisX + 2;
			ux[getFaceIndexX(ix, iy, iz)] = 0.0;
			ix = numberOfCellsPerAxisX + 1;
			ux[getFaceIndexX(ix, iy, iz)] = 0.0;

			ix = numberOfCellsPerAxisX + 1;
			uy[getFaceIndexY(ix, iy, iz)] = (UseNoSlip ? -1.0 : 1.0) * uy[getFaceIndexY(ix - 1, iy, iz)];
			uz[getFaceIndexZ(ix, iy, iz)] = (UseNoSlip ? -1.0 : 1.0) * uz[getFaceIndexZ(ix - 1, iy, iz)];
		}
	}

	for (ix = 0; ix<numberOfCellsPerAxisX + 2; ix++) {
		for (iz = 0; iz<numberOfCellsPerAxisZ + 2; iz++) {
			iy = 0;
			uy[getFaceIndexY(ix, iy, iz)] = 0.0;
			iy = 1;
			uy[getFaceIndexY(ix, iy, iz)] = 0.0;

			iy = 0;
			ux[getFaceIndexX(ix, iy, iz)] = (UseNoSlip ? -1.0 : 1.0) * ux[getFaceIndexX(ix, iy + 1, iz)];
			uz[getFaceIndexZ(ix, iy, iz)] = (UseNoSlip ? -1.0 : 1.0) * uz[getFaceIndexZ(ix, iy + 1, iz)];

			iy = numberOfCellsPerAxisY + 2;
			uy[getFaceIndexY(ix, iy, iz)] = 0.0;
			iy = numberOfCellsPerAxisY + 1;
			uy[getFaceIndexY(ix, iy, iz)] = 0.0;

			iy = numberOfCellsPerAxisY + 1;
			ux[getFaceIndexX(ix, iy, iz)] = (UseNoSlip ? -1.0 : 1.0) * ux[getFaceIndexX(ix, iy - 1, iz)];
			uz[getFaceIndexZ(ix, iy, iz)] = (UseNoSlip ? -1.0 : 1.0) * uz[getFaceIndexZ(ix, iy - 1, iz)];
		}
	}

	for (ix = 0; ix<numberOfCellsPerAxisX + 2; ix++) {
		for (iy = 0; iy<numberOfCellsPerAxisY + 2; iy++) {
			iz = 0;
			uz[getFaceIndexZ(ix, iy, iz)] = 0.0;
			iz = 1;
			uz[getFaceIndexZ(ix, iy, iz)] = 0.0;

			iz = 0;
			ux[getFaceIndexX(ix, iy, iz)] = (UseNoSlip ? -1.0 : 1.0) * ux[getFaceIndexX(ix, iy, iz + 1)];
			uy[getFaceIndexY(ix, iy, iz)] = (UseNoSlip ? -1.0 : 1.0) * uy[getFaceIndexY(ix, iy, iz + 1)];

			iz = numberOfCellsPerAxisZ + 2;
			uz[getFaceIndexZ(ix, iy, iz)] = 0.0;
			iz = numberOfCellsPerAxisZ + 1;
			uz[getFaceIndexZ(ix, iy, iz)] = 0.0;

			iz = numberOfCellsPerAxisZ + 1;
			ux[getFaceIndexX(ix, iy, iz)] = (UseNoSlip ? -1.0 : 1.0) * ux[getFaceIndexX(ix, iy, iz - 1)];
			uy[getFaceIndexY(ix, iy, iz)] = (UseNoSlip ? -1.0 : 1.0) * uy[getFaceIndexY(ix, iy, iz - 1)];
		}
	}

	validateThatEntriesAreBounded("setVelocityBoundaryConditions(double)[no slip set]");


	// Don't switch on in-flow immediately but slowly induce it into the system
	//
	const double drivenCavityVelocity = std::min(time, 1.0);

	for (iy = 1; iy<numberOfCellsPerAxisY + 2 - 1; iy++) {
		for (iz = 1; iz<numberOfCellsPerAxisZ + 2 - 1; iz++) {
			/*      ix=0;
			ux[ getFaceIndexX(ix,iy,iz) ] = 0.0;
			uy[ getFaceIndexY(ix,iy,iz) ] = drivenCavityVelocity;
			uz[ getFaceIndexZ(ix,iy,iz) ] = 0.0;
			ix=1;
			ux[ getFaceIndexX(ix,iy,iz) ] = 0.0;
			uy[ getFaceIndexY(ix,iy,iz) ] = drivenCavityVelocity;
			uz[ getFaceIndexZ(ix,iy,iz) ] = 0.0; */
			ix = 0;
			ux[getFaceIndexX(ix, iy, iz)] = 0.0;
			uy[getFaceIndexY(ix, iy, iz)] = drivenCavityVelocity - 1.0 * uy[getFaceIndexY(ix + 1, iy, iz)];
			uz[getFaceIndexZ(ix, iy, iz)] = -1.0 * uz[getFaceIndexZ(ix + 1, iy, iz)];
		}
	}

	validateThatEntriesAreBounded("setVelocityBoundaryConditions(double)[inflow set]");

	//
	//  Once all velocity boundary conditions are set, me can fix the F values
	//
	for (iy = 0; iy<numberOfCellsPerAxisY + 2; iy++) {
		for (iz = 0; iz<numberOfCellsPerAxisZ + 2; iz++) {
			ix = 0;
			Fx[getFaceIndexX(ix, iy, iz)] = ux[getFaceIndexX(ix, iy, iz)];

			Fy[getFaceIndexY(ix, iy, iz)] = uy[getFaceIndexY(ix, iy, iz)];
			Fz[getFaceIndexZ(ix, iy, iz)] = uz[getFaceIndexZ(ix, iy, iz)];
			//Fy[ getFaceIndex^(ix,iy,iz) ] = uz[ getFaceIndexZ(ix,iy,iz) ];
			//Fz[ getFaceIndexZ(ix,iy,iz) ] = uy[ getFaceIndexY(ix,iy,iz) ];

			ix = 1;
			Fx[getFaceIndexX(ix, iy, iz)] = ux[getFaceIndexX(ix, iy, iz)];

			ix = numberOfCellsPerAxisX + 1;
			Fx[getFaceIndexX(ix, iy, iz)] = ux[getFaceIndexX(ix, iy, iz)];
			Fy[getFaceIndexY(ix, iy, iz)] = uy[getFaceIndexY(ix, iy, iz)];
			Fz[getFaceIndexZ(ix, iy, iz)] = uz[getFaceIndexZ(ix, iy, iz)];
			//Fy[ getFaceIndexY(ix,iy,iz) ] = uz[ getFaceIndexY(ix,iy,iz) ];
			//Fz[ getFaceIndexZ(ix,iy,iz) ] = uy[ getFaceIndexZ(ix,iy,iz) ];

			ix = numberOfCellsPerAxisX + 2;
			Fx[getFaceIndexX(ix, iy, iz)] = ux[getFaceIndexX(ix, iy, iz)];
		}
	}

	for (ix = 0; ix<numberOfCellsPerAxisX + 2; ix++) {
		for (iz = 0; iz<numberOfCellsPerAxisZ + 2; iz++) {
			iy = 0;
			Fy[getFaceIndexY(ix, iy, iz)] = uy[getFaceIndexY(ix, iy, iz)];

			Fx[getFaceIndexX(ix, iy, iz)] = ux[getFaceIndexX(ix, iy, iz)];
			Fz[getFaceIndexZ(ix, iy, iz)] = uz[getFaceIndexZ(ix, iy, iz)];
			//Fx[ getFaceIndexX(ix,iy,iz) ] = uz[ getFaceIndexZ(ix,iy,iz) ];
			//Fz[ getFaceIndexZ(ix,iy,iz) ] = ux[ getFaceIndexX(ix,iy,iz) ];

			iy = 1;
			Fy[getFaceIndexY(ix, iy, iz)] = uy[getFaceIndexY(ix, iy, iz)]
				;
			iy = numberOfCellsPerAxisY + 1;
			Fy[getFaceIndexY(ix, iy, iz)] = uy[getFaceIndexY(ix, iy, iz)];
			Fx[getFaceIndexX(ix, iy, iz)] = ux[getFaceIndexX(ix, iy, iz)];
			Fz[getFaceIndexZ(ix, iy, iz)] = uz[getFaceIndexZ(ix, iy, iz)];
			//Fx[ getFaceIndexX(ix,iy,iz) ] = uz[ getFaceIndexZ(ix,iy,iz) ];
			//Fz[ getFaceIndexZ(ix,iy,iz) ] = ux[ getFaceIndexX(ix,iy,iz) ];

			iy = numberOfCellsPerAxisY + 2;
			Fy[getFaceIndexY(ix, iy, iz)] = uy[getFaceIndexY(ix, iy, iz)];
		}
	}


	for (ix = 0; ix<numberOfCellsPerAxisX + 2; ix++) {
		for (iy = 0; iy<numberOfCellsPerAxisY + 2; iy++) {
			iz = 0;
			Fz[getFaceIndexZ(ix, iy, iz)] = uz[getFaceIndexZ(ix, iy, iz)];

			Fx[getFaceIndexX(ix, iy, iz)] = ux[getFaceIndexX(ix, iy, iz)];
			Fy[getFaceIndexY(ix, iy, iz)] = uy[getFaceIndexY(ix, iy, iz)];
			//Fx[ getFaceIndexX(ix,iy,iz) ] = uy[ getFaceIndexY(ix,iy,iz) ];
			//Fy[ getFaceIndexY(ix,iy,iz) ] = ux[ getFaceIndexX(ix,iy,iz) ];

			iz = 1;
			Fz[getFaceIndexZ(ix, iy, iz)] = uz[getFaceIndexZ(ix, iy, iz)];

			iz = numberOfCellsPerAxisZ + 1;
			Fz[getFaceIndexZ(ix, iy, iz)] = uz[getFaceIndexZ(ix, iy, iz)];
			Fx[getFaceIndexX(ix, iy, iz)] = ux[getFaceIndexX(ix, iy, iz)];
			Fy[getFaceIndexY(ix, iy, iz)] = uy[getFaceIndexY(ix, iy, iz)];
			//Fx[ getFaceIndexX(ix,iy,iz) ] = uy[ getFaceIndexY(ix,iy,iz) ];
			//Fy[ getFaceIndexY(ix,iy,iz) ] = ux[ getFaceIndexX(ix,iy,iz) ];

			iz = numberOfCellsPerAxisZ + 2;
			Fz[getFaceIndexZ(ix, iy, iz)] = uz[getFaceIndexZ(ix, iy, iz)];
		}
	}

	validateThatEntriesAreBounded("setVelocityBoundaryConditions(double)[out]");
}

class RenderCommand : public vtkCommand
{
public:
	static RenderCommand * New()
	{
		return new RenderCommand;
	}

	void Execute(vtkObject * caller,
		unsigned long eventId,
		void * callData)
	{
		static_cast<vtkRenderWindowInteractor *> (caller)->Render();
	}
};

void interact(vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor) {
	// ParaView style
	vtkSmartPointer<vtkInteractorStyleTrackballCamera> style =
		vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();

	renderWindowInteractor->SetInteractorStyle(style);
	
	renderWindowInteractor->Initialize();
	
	// 30 FPS
	renderWindowInteractor->CreateRepeatingTimer(30);
	RenderCommand * renderCommand = RenderCommand::New();
	renderWindowInteractor->AddObserver(vtkCommand::TimerEvent, renderCommand);

	renderWindowInteractor->Start();
}

int main(int argc, char *argv[]) {
	if (argc != 4) {
		std::cout << "Usage: executable number-of-elements-per-axis time-steps-between-plots reynolds-number scenario" << std::endl;
		std::cout << "    number-of-elements-per-axis  Resolution. Start with 20, but try to increase as much as possible later." << std::endl;
		std::cout << "    time-between-plots           Determines how many files are written. Set to 0 to switch off plotting (for performance studies)." << std::endl;
		std::cout << "    reynolds-number              Use something in-between 1 and 1000. Determines viscosity of fluid." << std::endl;
		//std::cout << "    plane						   The plane to cut. [x,y or z]." << std::endl;
		return 1;
	}

	numberOfCellsPerAxisX = atoi(argv[1]);
	numberOfCellsPerAxisY = atoi(argv[1]);
	numberOfCellsPerAxisZ = atoi(argv[1]);
	double timeBetweenPlots = atof(argv[2]);
	ReynoldsNumber = atof(argv[3]);
	//char plane = atof(argv[4]);

	std::cout << "Re=" << ReynoldsNumber << std::endl;

	std::cout << "create " << numberOfCellsPerAxisX << "x" << numberOfCellsPerAxisY << "x" << numberOfCellsPerAxisZ << " grid" << std::endl;
	setupScenario();

	timeStepSize = 1e-2;
	std::cout << "start with time step size " << timeStepSize << std::endl;

	setVelocityBoundaryConditions(0.0);
	std::cout << "velocity start condition are set" << std::endl;

	/**
	* Vtk data
	*/
	vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
	vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	vtkSmartPointer<vtkRenderer> renderer;

	if (timeBetweenPlots>0.0) {

		// Get initial conditions
		renderer = makeGridImageData(imageData);
		renderWindow->AddRenderer(renderer);
		renderWindowInteractor->SetRenderWindow(renderWindow);
		std::cout << "start condition is calculated" << std::endl;
	}

	// Start event loop in new thread
	std::thread t1(interact, renderWindowInteractor);

	double t = 0.0;
	double tOfLastSnapshot = 0.0;
	int    timeStepCounter = 0;
	while (t<TerminalTime) {
		std::cout << "time step " << timeStepCounter << ": t=" << t << "\t dt=" << timeStepSize << "\t";

		setVelocityBoundaryConditions(t);
		computeF();
		computeRhs();
		int innerIterationsRequired = computeP();
		setNewVelocities();

		if (timeBetweenPlots>0.0 && (t - tOfLastSnapshot>timeBetweenPlots)) {
			updateVisualisation(imageData);
			tOfLastSnapshot = t;
		}

		if (innerIterationsRequired == MaxComputePIterations && timeStepSize>1e-10) {
			timeStepSize /= 2.0;
			std::cout << "\t time step size seems to be too big. Reduced to " << timeStepSize << std::endl;
		}
		if (
			innerIterationsRequired <= MinAverageComputePIterations
			&&
			timeStepSize < 0.9 * ReynoldsNumber * 1.0 / static_cast<double>(numberOfCellsPerAxisX) * 1.0 / static_cast<double>(numberOfCellsPerAxisX)
			) {
			timeStepSize *= (1.0 + ChangeOfTimeStepSize);
			std::cout << "\t time step size seems to be too small. Increased to " << timeStepSize << std::endl;
		}

		t += timeStepSize;
		timeStepCounter++;

		std::cout << std::endl;
	}

	std::cout << "free data structures" << std::endl;
	freeDataStructures();

	t1.join();

	return 0;
}