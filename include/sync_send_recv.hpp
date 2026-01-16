#pragma once

#include <utils.hpp>
#include <mpi.h>

#include "message_layout.hpp"
#include "packed_message.hpp"

namespace mpi {

void sync_send(const int dest, const PackedMessage& message);

int sync_recv(PackedMessage& message, int source = MPI_ANY_SOURCE);

}
