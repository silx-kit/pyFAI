from libcpp cimport bool
cdef extern from "ocl_xrpd1d.hpp":
    cdef cppclass ocl_xrpd1D_fullsplit:
        # Default constructor - Prints messages on stdout
        ocl_xrpd1D_fullsplit()

        # other Constructor: Prints messages on file fname
        ocl_xrpd1D_fullsplit(char * fname)

        # getConfiguration gets the description of the integrations to be performed and keeps an internal copy
        int getConfiguration(int Nx, int Nimage, int Nbins, bool usefp64) nogil

        # configure is possibly the most crucial method of the class.
        # It is responsible of allocating the required memory and compile the OpenCL kernels
        # based on the configuration of the integration.
        # It also "ties" the OpenCL memory to the kernel arguments.
        # If ANY of the arguments of getConfiguration needs to be changed, configure must
        # be called again for them to take effect
        int configure(char * kernel_path) nogil

        # Load the 2th arrays along with the min and max value.
        # loadTth maybe be recalled at any time of the execution in order to update
        # the 2th arrays.
        #
        # loadTth is required and must be called at least once after a configure()
        int loadTth(float * tth, float * dtth, float tth_min, float tth_max) nogil

        # Enables SolidAngle correction and uploads the suitable array to the OpenCL device.
        # By default the program will assume no solidangle correction unless setSolidAngle() is called.
        # From then on, all integrations will be corrected via the SolidAngle array.
        #
        # If the SolidAngle array needs to be changes, one may just call setSolidAngle() again
        # with that array
        int setSolidAngle(float * SolidAngle) nogil

        # Instructs the program to not perform solidangle correction from now on.
        # SolidAngle correction may be turned back on at any point
        int unsetSolidAngle() nogil

        # Enables the use of a Mask during integration. The Mask can be updated by
        # recalling setMask at any point.
        #
        # The Mask must be a PyFAI Mask
        int setMask(int * Mask) nogil

        # Disables the use of a Mask from that point. It may be reenabled at any point
        # via setMask
        int unsetMask() nogil

        # Enables dummy value functionality and uploads the value to the OpenCL device.
        # Image values that are similar to the dummy value are set to 0.
        int setDummyValue(float dummyVal, float deltaDummyVal) nogil

        # Disable a dummy value. May be reenabled at any time by setDummyValue
        int unsetDummyValue() nogil

        # Sets the active range to integrate on. By default the range is set to tth_min and tth_max
        # By calling this functions, one may change to different bounds
        int setRange(float lowerBound, float upperBound) nogil

        # Resets the 2th integration range back to tth_min, tth_max
        int unsetRange() nogil

        # Take an image, integrate and return the histogram and weights
        # set / unset and loadTth methods have a direct impact on the execute() method.
        # All the rest of the methods will require at least a new configuration via configure()
        int execute(float * im_inten, float * histogram, float * bins) nogil

        # Free OpenCL related resources.
        # It may be asked to preserve the context created by init or completely clean up OpenCL.
        #
        # Guard / Status flags that are set will be reset. All the Operation flags are also reset
        int clean(int preserve_context) nogil


        ################################################################################
        # Inherited from ocl_base
        ################################################################################

        #Initial configuration: Choose a device and initiate a context. Devicetypes can be GPU,gpu,CPU,cpu,DEF,ACC,ALL.
        #Suggested are GPU,CPU. For each setting to work there must be such an OpenCL device and properly installed.
        #E.g.: If Nvidia driver is installed, GPU will succeed but CPU will fail. The AMD SDK kit is required for CPU via OpenCL.
        int init(bool useFp64) nogil
        int init(char * devicetype, bool useFp64) nogil
        int init(char * devicetype, int platformid, int devid, bool useFp64) nogil


        #Prints a list of OpenCL capable devices, their platforms and their ids
        void show_devices() nogil


        #Same as show_devices but displays the results always on stdout even
        #if the stream is set to a file
        void print_devices() nogil


        #Print details of a selected device
        void show_device_details() nogil

        #Resets the internal profiling timers to 0
        void  reset_time() nogil

        #Returns the internal profiling timer for the kernel executions
        float get_exec_time() nogil

        #Returns how many integrations have been performed
        unsigned int get_exec_count() nogil

        #Returns the time spent on memory copies
        float get_memCpy_time() nogil

        int get_status() nogil

        void print_active_platform_info() nogil

        void print_active_device_info() nogil

        void return_pair(int & platform, int & device) nogil
