# file: cqueue.pxd

cdef extern from "sys/queue.h":
    "Singly-linked List definitions."
    ctypedef struct TYPE:
        pass
    ctypedef struct HEADNAME:
        pass
    bint SLIST_HEAD(HEADNAME, TYPE)
    bint SLIST_EMPTY(SLIST_HEAD * head)
    bint SLIST_ENTRY(TYPE)
    TYPE SLIST_FIRST(SLIST_HEAD * head)
    TYPE SLIST_FOREACH(TYPE * var, SLIST_HEAD * head, SLIST_ENTRY NAME)
    bint SLIST_FOREACH_SAFE(TYPE * var, SLIST_HEAD * head, SLIST_ENTRY NAME,
        TYPE * temp_var)
    bint SLIST_HEAD_INITIALIZER(SLIST_HEAD head)
    bint SLIST_INIT(SLIST_HEAD * head)
    bint SLIST_INSERT_AFTER(TYPE * listelm, TYPE * elm, SLIST_ENTRY NAME)
    bint SLIST_INSERT_HEAD(SLIST_HEAD * head, TYPE * elm, SLIST_ENTRY NAME)
    TYPE SLIST_NEXT(TYPE * elm, SLIST_ENTRY NAME)
    bint SLIST_REMOVE_HEAD(SLIST_HEAD * head, SLIST_ENTRY NAME
    bint SLIST_REMOVE(SLIST_HEAD * head, TYPE * elm, TYPE, SLIST_ENTRY NAME)
