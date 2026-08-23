"""Dense kernels for category encoders, exposed through a C ABI."""

from std.sys import simd_width_of

comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]


@always_inline
def rotl32(value: UInt32, bits: UInt32) -> UInt32:
    return (value << bits) | (value >> (UInt32(32) - bits))


@always_inline
def padded_byte(data: BPtr, start: Int, n: Int, pos: Int, total: Int) -> UInt8:
    if pos < n:
        return data[unsafe_offset=start + pos]
    if pos == n:
        return UInt8(0x80)
    if pos >= total - 8:
        var bits = UInt64(n) * 8
        return UInt8(bits >> UInt64(8 * (pos - (total - 8))))
    return UInt8(0)


def md5_bucket(
    data: BPtr,
    start: Int,
    n: Int,
    components: Int,
    shifts: U32Ptr,
    constants: U32Ptr,
) -> Int:
    var a0 = UInt32(0x67452301)
    var b0 = UInt32(0xEFCDAB89)
    var c0 = UInt32(0x98BADCFE)
    var d0 = UInt32(0x10325476)
    var total = ((n + 1 + 8 + 63) // 64) * 64

    for block in range(total // 64):
        var base = block * 64
        var a = a0
        var b = b0
        var c = c0
        var d = d0

        for i in range(64):
            var f: UInt32
            var g: Int
            if i < 16:
                f = (b & c) | ((~b) & d)
                g = i
            elif i < 32:
                f = (d & b) | ((~d) & c)
                g = (5 * i + 1) % 16
            elif i < 48:
                f = b ^ c ^ d
                g = (3 * i + 5) % 16
            else:
                f = c ^ (b | (~d))
                g = (7 * i) % 16

            var word_pos = base + g * 4
            var word = (
                UInt32(padded_byte(data, start, n, word_pos, total))
                | (UInt32(padded_byte(data, start, n, word_pos + 1, total)) << 8)
                | (UInt32(padded_byte(data, start, n, word_pos + 2, total)) << 16)
                | (UInt32(padded_byte(data, start, n, word_pos + 3, total)) << 24)
            )
            var previous_d = d
            d = c
            c = b
            b += rotl32(
                a + f + constants[unsafe_offset=i] + word,
                shifts[unsafe_offset=i],
            )
            a = previous_d

        a0 += a
        b0 += b
        c0 += c
        d0 += d

    var remainder = 0
    for word_i in range(4):
        var word = a0
        if word_i == 1:
            word = b0
        elif word_i == 2:
            word = c0
        elif word_i == 3:
            word = d0
        for byte_i in range(4):
            var byte = UInt8(word >> UInt32(byte_i * 8))
            remainder = (remainder * 256 + Int(byte)) % components
    return remainder


@export("mce_hash_md5_buckets")
def mce_hash_md5_buckets(
    data_addr: Int,
    starts_addr: Int,
    lengths_addr: Int,
    shifts_addr: Int,
    constants_addr: Int,
    buckets_addr: Int,
    count: Int,
    components: Int,
) abi("C"):
    var data = BPtr(unsafe_from_address=data_addr)
    var starts = IPtr(unsafe_from_address=starts_addr)
    var lengths = IPtr(unsafe_from_address=lengths_addr)
    var shifts = U32Ptr(unsafe_from_address=shifts_addr)
    var constants = U32Ptr(unsafe_from_address=constants_addr)
    var buckets = IPtr(unsafe_from_address=buckets_addr)

    @__parameter
    def hash_chunk(chunk: Int):
        var begin = chunk * 256
        var end = min(begin + 256, count)
        for i in range(begin, end):
            buckets[unsafe_offset=i] = Int64(md5_bucket(
                data,
                Int(starts[unsafe_offset=i]),
                Int(lengths[unsafe_offset=i]),
                components,
                shifts,
                constants,
            ))

    for chunk in range((count + 255) // 256):
        hash_chunk(chunk)


@always_inline
def zero_i64(result: IPtr, n: Int):
    comptime W = simd_width_of[DType.int64]()
    var i = 0
    var zeros = SIMD[DType.int64, W](0)
    while i + W <= n:
        result.unsafe_store(i, zeros)
        i += W
    while i < n:
        result[unsafe_offset=i] = 0
        i += 1


@export("mce_hash_accumulate")
def mce_hash_accumulate(
    codes_addr: Int,
    buckets_addr: Int,
    result_addr: Int,
    rows: Int,
    columns: Int,
    components: Int,
) abi("C"):
    var codes = IPtr(unsafe_from_address=codes_addr)
    var buckets = IPtr(unsafe_from_address=buckets_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    zero_i64(result, rows * components)

    @__parameter
    def accumulate_row(row: Int):
        var code_base = row * columns
        var result_base = row * components
        for column in range(columns):
            var code = Int(codes[unsafe_offset=code_base + column])
            if code >= 0:
                result[
                    unsafe_offset=result_base + Int(buckets[unsafe_offset=code])
                ] += 1

    for row in range(rows):
        accumulate_row(row)


@export("mce_ordinal_apply")
def mce_ordinal_apply(
    positions_addr: Int,
    lookup_addr: Int,
    result_addr: Int,
    n: Int,
    lookup_len: Int,
    unknown_value: Float64,
    missing_value: Float64,
) abi("C"):
    var positions = IPtr(unsafe_from_address=positions_addr)
    var lookup = FPtr(unsafe_from_address=lookup_addr)
    var result = FPtr(unsafe_from_address=result_addr)
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        var position_vec = positions.unsafe_load[width=W](i)
        var result_vec = SIMD[DType.float64, W](0.0)
        comptime for lane in range(W):
            var code = Int(position_vec[lane])
            if code >= 0 and code < lookup_len:
                result_vec[lane] = lookup[unsafe_offset=code]
            elif code == -2:
                result_vec[lane] = missing_value
            else:
                result_vec[lane] = unknown_value
        result.unsafe_store(i, result_vec)
        i += W
    while i < n:
        var code = Int(positions[unsafe_offset=i])
        if code >= 0 and code < lookup_len:
            result[unsafe_offset=i] = lookup[unsafe_offset=code]
        elif code == -2:
            result[unsafe_offset=i] = missing_value
        else:
            result[unsafe_offset=i] = unknown_value
        i += 1


@export("mce_target_stats")
def mce_target_stats(
    codes_addr: Int,
    target_addr: Int,
    counts_addr: Int,
    sums_addr: Int,
    n: Int,
    groups: Int,
) abi("C"):
    var codes = IPtr(unsafe_from_address=codes_addr)
    var target = FPtr(unsafe_from_address=target_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)
    var sums = FPtr(unsafe_from_address=sums_addr)
    for i in range(groups):
        counts[unsafe_offset=i] = 0
        sums[unsafe_offset=i] = 0.0
    for i in range(n):
        var group = Int(codes[unsafe_offset=i]) + 2
        if group >= 0 and group < groups:
            counts[unsafe_offset=group] += 1
            sums[unsafe_offset=group] += target[unsafe_offset=i]


@export("mce_target_apply")
def mce_target_apply(
    codes_addr: Int,
    mapping_addr: Int,
    result_addr: Int,
    n: Int,
    mapping_len: Int,
    default_value: Float64,
) abi("C"):
    var codes = IPtr(unsafe_from_address=codes_addr)
    var mapping = FPtr(unsafe_from_address=mapping_addr)
    var result = FPtr(unsafe_from_address=result_addr)

    @__parameter
    def apply_chunk(chunk: Int):
        comptime W = simd_width_of[DType.float64]()
        var begin = chunk * 65536
        var end = min(begin + 65536, n)
        var i = begin
        while i + W <= end:
            var code_vec = codes.unsafe_load[width=W](i)
            var result_vec = SIMD[DType.float64, W](default_value)
            comptime for lane in range(W):
                var group = Int(code_vec[lane]) + 2
                if group >= 0 and group < mapping_len:
                    result_vec[lane] = mapping[unsafe_offset=group]
            result.unsafe_store(i, result_vec)
            i += W
        while i < end:
            var group = Int(codes[unsafe_offset=i]) + 2
            result[unsafe_offset=i] = (
                mapping[unsafe_offset=group]
                if group >= 0 and group < mapping_len
                else default_value
            )
            i += 1

    for chunk in range((n + 65535) // 65536):
        apply_chunk(chunk)
