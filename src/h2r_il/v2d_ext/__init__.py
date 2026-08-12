"""Modules that run *inside* v2d's docker images, not in this environment.

They are bind-mounted as single files into the image's existing ``/workspace``
rather than dev-mounting the whole modules tree, because the images carry
compiled CUDA extensions under that path which a full mount would hide. That
also keeps the shared ``video_to_data`` checkout a read-only dependency: nothing
here is copied into it.

Do not import these from host code -- they depend on packages that only exist
inside the containers.
"""
