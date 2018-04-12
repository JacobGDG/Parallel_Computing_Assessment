#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include<cmath>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

typedef struct my_struct {
	string location;
	int year;
	int month;
	int day;
	string time;
	float temp;
}my_struct;

//cout for my_struct
ostream &operator<<(std::ostream &os, my_struct const &m) {
	os << "Location: " << m.location << " Date: " << m.year << "/" << m.month << "/" << m.day << ", Time: " << m.time << ", Temp(c): " << m.temp;
	return os;
}

const string fileLocationShort = "temp_lincolnshire_short.txt", fileLocationLong = "temp_lincolnshire.txt";

typedef int mytype;
vector<mytype> test;

vector<my_struct> rawData;
ifstream tempsFile;

float minTemp, maxTemp, meanTemp;

//open fileLocation. Read each line then save each word into rawData vector.
bool ReadDataFromTXT(bool fullData)
{
	string line, fileLocation;
	int count = 0;

	fileLocation = fullData ? fileLocationLong : fileLocationShort;

	tempsFile.open(fileLocation);//open file

	if (tempsFile)//check file is open
	{
		cout << "Reading data to array..." << endl << endl;

		while (getline(tempsFile, line))//read each line of file into string
		{
			stringstream linestream(line);//store string into stringstream
			my_struct temp;

			linestream >> temp.location >>
				(int)temp.year >>
				(int)temp.month >>
				(int)temp.day >>
				temp.time >>
				(float)temp.temp;//store each value in stream

			test.push_back(floor(temp.temp * 10));
			rawData.push_back(temp);

			if (count % 100000 == 0)
				cout << "Progress: " << count << endl;// << "/" << maxCount << endl;

			count++;//next object
		}

		tempsFile.close();//close file

		cout << "Progress: " << count << "/" << count << endl;
		cout << "File input successful! Beggining multicore calculations..." << endl << endl;

		return true;
	}
	else//if file does not exist
	{
		cout << "there was an error opening: " << fileLocation << endl;
		return false;
	}
}

int main(int argc, char **argv) 
{	
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
		
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}
		
	//detect any potential exceptions
	try 
	{
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);
		
		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		
		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);
		
		//2.2 Load & build the device code
		cl::Program::Sources sources;
		
		AddSources(sources, "my_kernels3.cl");
		
		cl::Program program(context, sources);
		
		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		if (ReadDataFromTXT(false))//read file data. skip rest if not exist
		{


			//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
			//if the total input length is divisible by the workgroup size
			//this makes the code more efficient

			size_t local_size = 10;

			size_t padding_size = test.size() % local_size;

			//if the input vector is not a multiple of the local_size
			//insert additional neutral elements (0 for addition) so that the total will not be affected
			if (padding_size) {
				//create an extra vector with neutral values
				std::vector<int> rawData_ext(local_size - padding_size, 0);
				//append that extra vector to our input
				test.insert(test.end(), rawData_ext.begin(), rawData_ext.end());
			}
			test[100];
			size_t input_elements = test.size();//number of input elements
			size_t input_size = test.size()*sizeof(mytype);//size in bytes
			size_t nr_groups = input_elements / local_size;

			//host - output
			std::vector<mytype> B(input_elements);
			size_t output_size = B.size()*sizeof(mytype);//size in bytes

			//device - buffers
			cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
			cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);


			//Part 5 - device operations

			//5.1 copy array A to and initialise other arrays on device memory
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &test[0]);
			queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

			//5.2 Setup and execute all kernels (i.e. device code)
			cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_4");
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size


			//call all kernels in a sequence
			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
			//5.3 Copy the result from device to host
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
			meanTemp = (static_cast<float>(B[0]) / (float)10) / test.size();

			queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

			kernel_1 = cl::Kernel(program, "reduce_min");
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

			minTemp = static_cast<float>(B[0]) / (float)10;

			queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

			kernel_1 = cl::Kernel(program, "reduce_max");
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

			maxTemp = static_cast<float>(B[0]) / (float)10;

			//std::cout << "A = " << A << std::endl;
			cout << fixed;
			cout.precision(1);
			std::cout << "Average Temperature (Celcius) = " << meanTemp << std::endl;
			std::cout << "Min Temperature = " << minTemp << endl;
			std::cout << "Max Temperature = " << maxTemp << endl;
		}
	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	//delete[] myRawData;//delete array;
	cout << endl << "Enter to exit.";
	cin.get();
	return 0;
}