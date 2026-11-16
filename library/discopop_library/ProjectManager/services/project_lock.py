# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2020, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# the 3-Clause BSD License.  See the LICENSE file in the package base
# directory for details.

import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass(frozen=True)
class LockHolder:
    owner: str
    since: float


class ProjectLock:
    def __init__(self, on_change: Callable[[Optional[LockHolder]], None]):
        self._on_change: Callable[[Optional[LockHolder]], None] = on_change
        self._holder: LockHolder = None
        self._mutex: threading.Lock = threading.Lock()

    def try_acquire(self, owner: str) -> Optional[LockHolder]:
        with self._mutex:
            if self._holder is not None:
                return self._holder
            self._holder = LockHolder(owner, time.monotonic())
            holder = self._holder
        self._on_change(holder) # outside the mutex: the callback must not run while holding it
        return None

    def release(self, owner: str) -> None:
        with self._mutex:
            if self._holder is None:
                return None
            if self._holder.owner != owner:
                raise Exception(f"Trying to release a project lock {str(self._holder)} that is not owned by {owner}")
            self._holder = None
        self._on_change(None) # outside the mutex: the callback must not run while holding it

    @property
    def holder(self) -> Optional[LockHolder]:
        return self._holder