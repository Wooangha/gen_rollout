from typing import TypeGuard, cast

from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch as th

from gymnasium.spaces import Space, Dict, Tuple, Box

ObsBuffer = list[np.ndarray] | dict[str, "ObsBuffer"] | tuple["ObsBuffer", ...]
ObsBufferArray = np.ndarray | dict[str, "ObsBufferArray"] | tuple["ObsBufferArray", ...]
ObsType = np.ndarray | dict[str, "ObsType"] | tuple["ObsType", ...]


def create_obs_buffer(space: Space) -> ObsBuffer:
    """Create an observation buffer based on the observation space."""
    if isinstance(space, Dict):
        return {key: create_obs_buffer(subspace) for key, subspace in space.spaces.items()}
    elif isinstance(space, Tuple):
        return tuple(create_obs_buffer(subspace) for subspace in space.spaces)
    else:
        return []


def append_to_obs_buffer(buffer: ObsBuffer, obs: ObsType) -> None:
    """Append an observation to the observation buffer."""
    if isinstance(buffer, dict) and isinstance(obs, dict):
        for key in buffer:
            append_to_obs_buffer(buffer[key], obs[key])
    elif isinstance(buffer, tuple) and isinstance(obs, tuple):
        for buf, ob in zip(buffer, obs):
            append_to_obs_buffer(buf, ob)
    elif isinstance(buffer, list) and isinstance(obs, np.ndarray):
        buffer.append(obs)
    elif isinstance(buffer, list) and isinstance(obs, (int, float)):
        buffer.append(np.array(obs))
    else:
        raise ValueError("Mismatch between buffer and observation types.")


def obs_buffer_to_array(buffer: ObsBuffer) -> ObsBufferArray:
    """Convert an observation buffer to a numpy array."""
    if isinstance(buffer, dict):
        return {key: obs_buffer_to_array(subbuffer) for key, subbuffer in buffer.items()}
    elif isinstance(buffer, tuple):
        return tuple(obs_buffer_to_array(subbuffer) for subbuffer in buffer)
    else:
        return np.stack(buffer)


ObsArrayBytes = int | dict[str, "ObsArrayBytes"] | tuple["ObsArrayBytes", ...]


def get_obs_bytes(buffer: ObsBufferArray) -> ObsArrayBytes:
    """Calculate the total number of bytes used by the observation array."""
    if isinstance(buffer, dict):
        return {key: get_obs_bytes(subbuffer) for key, subbuffer in buffer.items()}
    elif isinstance(buffer, tuple):
        return tuple(get_obs_bytes(subbuffer) for subbuffer in buffer)
    else:
        return buffer.nbytes


Shape = tuple[int, ...]
ObsArrayShape = Shape | dict[str, "ObsArrayShape"] | tuple["ObsArrayShape", ...]


def is_shape(x) -> TypeGuard[Shape]:
    return isinstance(x, tuple) and len(x) > 0 and all(isinstance(i, int) for i in x)


def get_obs_shape(buffer: ObsBufferArray) -> ObsArrayShape:
    """Get the shape of the observation array."""
    if isinstance(buffer, dict):
        return {key: get_obs_shape(subbuffer) for key, subbuffer in buffer.items()}
    elif isinstance(buffer, tuple):
        return tuple(get_obs_shape(subbuffer) for subbuffer in buffer)
    else:
        return buffer.shape


ObsSharedMemory = SharedMemory | dict[str, "ObsSharedMemory"] | tuple["ObsSharedMemory", ...]


def make_shm_from_obs_array(bytes: ObsArrayBytes) -> ObsSharedMemory:
    """Create a SharedMemory object from an observation array."""
    if isinstance(bytes, dict):
        return {key: make_shm_from_obs_array(subbytes) for key, subbytes in bytes.items()}
    elif isinstance(bytes, tuple):
        return tuple(make_shm_from_obs_array(subbytes) for subbytes in bytes)
    else:
        return SharedMemory(create=True, size=bytes)


def make_obs_with_shm(buffer: ObsBufferArray, shm: ObsSharedMemory) -> ObsBufferArray:
    """Create an observation array that uses shared memory."""
    if isinstance(buffer, dict) and isinstance(shm, dict):
        return {key: make_obs_with_shm(buffer[key], shm[key]) for key in buffer}
    elif isinstance(buffer, tuple) and isinstance(shm, tuple):
        return tuple(make_obs_with_shm(buf, sh) for buf, sh in zip(buffer, shm))
    elif isinstance(buffer, np.ndarray) and isinstance(shm, SharedMemory):
        return np.ndarray(buffer.shape, dtype=buffer.dtype, buffer=shm.buf)
    else:
        raise ValueError("Mismatch between buffer and shared memory types.")


ObsDtype = np.dtype | dict[str, "ObsDtype"] | tuple["ObsDtype", ...]


def make_obs_with_shape_dtype_shm(shape: ObsArrayShape, dtype: ObsDtype, shm: ObsSharedMemory) -> ObsBufferArray:
    """Create an observation array with given shape and dtype that uses shared memory."""
    if is_shape(shape) and isinstance(dtype, np.dtype) and isinstance(shm, SharedMemory):
        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    elif isinstance(shape, dict) and isinstance(dtype, dict) and isinstance(shm, dict):
        return {key: make_obs_with_shape_dtype_shm(shape[key], dtype[key], shm[key]) for key in shape}
    elif isinstance(shape, tuple) and isinstance(dtype, tuple) and isinstance(shm, tuple):
        shape = cast(tuple[ObsArrayShape], shape)
        return tuple(make_obs_with_shape_dtype_shm(s, d, sh) for s, d, sh in zip(shape, dtype, shm))
    else:
        raise ValueError("Mismatch between shape, dtype, and shared memory types.")


def copy_obs(src: ObsBufferArray, dest: ObsBufferArray) -> None:
    """Copy an observation array to shared memory."""
    if isinstance(src, dict) and isinstance(dest, dict):
        for key in src:
            copy_obs(src[key], dest[key])
    elif isinstance(src, tuple) and isinstance(dest, tuple):
        for s, d in zip(src, dest):
            copy_obs(s, d)
    elif isinstance(src, np.ndarray) and isinstance(dest, np.ndarray):
        np.copyto(dest, src)
    else:
        raise ValueError("Mismatch between source and destination types.")


def close_obs_shm(shm: ObsSharedMemory):
    """Free the shared memory used by the observation array."""
    if isinstance(shm, dict):
        for subshm in shm.values():
            close_obs_shm(subshm)
    elif isinstance(shm, tuple):
        for subshm in shm:
            close_obs_shm(subshm)
    else:
        shm.close()


def unlink_obs_shm(shm: ObsSharedMemory):
    """Unlink the shared memory used by the observation array."""
    if isinstance(shm, dict):
        for subshm in shm.values():
            unlink_obs_shm(subshm)
    elif isinstance(shm, tuple):
        for subshm in shm:
            unlink_obs_shm(subshm)
    else:
        shm.unlink()


def slice_obs_array(
    obs_array: ObsBufferArray,
    start: int,
    end: int,
) -> ObsBufferArray:
    """Slice an observation array from start to end."""
    if isinstance(obs_array, dict):
        return {key: slice_obs_array(subarray, start, end) for key, subarray in obs_array.items()}
    elif isinstance(obs_array, tuple):
        return tuple(slice_obs_array(subarray, start, end) for subarray in obs_array)
    else:
        return obs_array[start:end]


ObsTensor = th.Tensor | dict[str, "ObsTensor"] | tuple["ObsTensor", ...]


def obs_to_tensor(obs: ObsBufferArray) -> ObsTensor:
    """Convert an observation array to a torch tensor."""
    if isinstance(obs, dict):
        return {key: obs_to_tensor(subobs) for key, subobs in obs.items()}
    elif isinstance(obs, tuple):
        return tuple(obs_to_tensor(subobs) for subobs in obs)
    else:
        return th.from_numpy(obs)


def obs_to_device(obs: ObsTensor, device: th.device) -> ObsTensor:
    """Move an observation tensor to a specified device."""
    if isinstance(obs, dict):
        return {key: obs_to_device(subobs, device) for key, subobs in obs.items()}
    elif isinstance(obs, tuple):
        return tuple(obs_to_device(subobs, device) for subobs in obs)
    else:
        return obs.to(device)


ObsShmName = str | dict[str, "ObsShmName"] | tuple["ObsShmName", ...]


def get_obs_shm_name(shm: ObsSharedMemory) -> ObsShmName:
    """Get the name of the shared memory used by the observation array."""
    if isinstance(shm, dict):
        return {key: get_obs_shm_name(subshm) for key, subshm in shm.items()}
    elif isinstance(shm, tuple):
        return tuple(get_obs_shm_name(subshm) for subshm in shm)
    else:
        return shm.name
