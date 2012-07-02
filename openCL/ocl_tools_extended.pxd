from libcpp cimport bool

cdef extern from "ocl_tools/ocl_tools_datatypes.h":
    # OpenCL tools platform information struct
    # 
    # It can be passed to ocl_platform_info to
    # retrieve and save platform information
    # for the current context
    ctypedef struct ocl_plat_t:
        char * name
        char * vendor
        char * version
        char * extensions
    #OpenCL tools platform information struct
    # It can be passed to ocl_device_info to
    # retrieve and save device information
    # for the current context
    ctypedef struct ocl_dev_t:
        char * name
        char type[4]
        char * version
        char * driver_version
        char * extensions
        unsigned long global_mem

#    OpenCL tools generic device information struct
#    Used to provide information for all the OpenCL devices
    ctypedef struct ocl_gen_dev_info_t:
        unsigned int Ndevices
        unsigned int * device_ids
        ocl_plat_t platform_info
        ocl_dev_t * device_info

#    OpenCL tools generic platform information struct
#    Chains information for each pair
    ctypedef struct ocl_gen_info_t:
        unsigned int Nplatforms
        unsigned int * platform_ids
        unsigned int ** ids #; //TODO
        ocl_gen_dev_info_t * platform
#    Returns info and pairs for all platforms and their devices
    ocl_gen_info_t * ocl_get_all_device_info(ocl_gen_info_t * Ninfo)


#    brief Releases all memory in Ninfo
    void ocl_clr_all_device_info(ocl_gen_info_t * Ninfo)
