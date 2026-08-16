from typing import TYPE_CHECKING

from exo.shared.types.common import NodeId
from exo.utils.pydantic_ext import FrozenModel

if TYPE_CHECKING:
    from exo_rs import FromSwarm

"""Serialisable types for Connection Updates/Messages"""


class ConnectionMessage(FrozenModel):
    connected: bool
    peer_id: NodeId

    @classmethod
    def from_update(
        cls, update: "FromSwarm.Connection"
    ) -> "ConnectionMessage":
        return cls(connected=update.connected, peer_id=NodeId(update.peer_id))
