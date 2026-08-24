#!/usr/bin/env bash
# probe_arch.sh: probe which Tensor Core PTX instructions ptxas accepts per arch (no GPU needed).
# GB10 is sm_121 (consumer Blackwell): wgmma (sm_90a) and tcgen05 (sm_100a) are unavailable.
set -uo pipefail

ARCHES="${ARCHES:-sm_90a sm_100a sm_121 sm_121f sm_121a}"
PTXAS="${PTXAS:-ptxas}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

command -v "$PTXAS" >/dev/null || { echo "ptxas not found. Load CUDA or set PTXAS=/path/to/ptxas"; exit 1; }

echo
"$PTXAS" --version | tail -2
echo

# sm_121 needs PTX ISA >= 8.8; find the minimum this ptxas accepts so probes fail only on instructions.
ISA=""
for v in 8.8 8.9 9.0 9.1 8.7; do
  printf '.version %s\n.target sm_121\n.address_size 64\n.visible .entry p(){ ret; }\n' "$v" > "$WORK/v.ptx"
  if "$PTXAS" -arch=sm_121 "$WORK/v.ptx" -o /dev/null 2>/dev/null; then ISA="$v"; break; fi
done
[ -z "$ISA" ] && { echo "No PTX ISA version in this ptxas accepts sm_121. This CUDA is too old for GB10 (need >= 12.9)."; exit 1; }
echo "using PTX ISA .version $ISA (minimum that accepts sm_121)"

emit() { cat > "$WORK/$1.ptx" <<EOF
.version $ISA
.target sm_121
.address_size 64
.visible .entry probe(.param .u64 p)
{
$2
    ret;
}
EOF
}

emit mma.sync.bf16 '
    .reg .b32 %ra<4>; .reg .b32 %rb<2>; .reg .f32 %rc<4>;
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%rc0,%rc1,%rc2,%rc3},{%ra0,%ra1,%ra2,%ra3},{%rb0,%rb1},{%rc0,%rc1,%rc2,%rc3};'
emit mma.sync.f16 '
    .reg .b32 %ra<4>; .reg .b32 %rb<2>; .reg .f32 %rc<4>;
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {%rc0,%rc1,%rc2,%rc3},{%ra0,%ra1,%ra2,%ra3},{%rb0,%rb1},{%rc0,%rc1,%rc2,%rc3};'
emit ldmatrix '
    .reg .b32 %r<4>; .reg .b64 %rd<2>; .reg .b32 %addr;
    ld.param.u64 %rd0,[p]; cvta.to.shared.u64 %rd1,%rd0; cvt.u32.u64 %addr,%rd1;
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0,%r1,%r2,%r3},[%addr];'
emit cp.async '
    .reg .b64 %rd<2>; .reg .b32 %addr;
    ld.param.u64 %rd0,[p]; cvta.to.shared.u64 %rd1,%rd0; cvt.u32.u64 %addr,%rd1;
    cp.async.ca.shared.global [%addr],[%rd0],16; cp.async.commit_group; cp.async.wait_group 0;'
emit wgmma.fence '
    wgmma.fence.sync.aligned;'
emit tcgen05.alloc '
    .reg .b64 %rd<2>; .reg .b32 %addr;
    ld.param.u64 %rd0,[p]; cvta.to.shared.u64 %rd1,%rd0; cvt.u32.u64 %addr,%rd1;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%addr],32;'
emit tma.bulk '
    .reg .b64 %rd<3>; .reg .b32 %addr,%mbar;
    ld.param.u64 %rd0,[p]; cvta.to.shared.u64 %rd1,%rd0; cvt.u32.u64 %addr,%rd1; mov.u32 %mbar,%addr;
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%addr],[%rd0],1024,[%mbar];'
emit tma.multicast '
    .reg .b64 %rd<3>; .reg .b32 %addr,%mbar; .reg .b16 %mask;
    ld.param.u64 %rd0,[p]; cvta.to.shared.u64 %rd1,%rd0; cvt.u32.u64 %addr,%rd1;
    mov.u32 %mbar,%addr; mov.u16 %mask,1;
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
        [%addr],[%rd0],1024,[%mbar],%mask;'

PROBES="mma.sync.bf16 mma.sync.f16 ldmatrix cp.async wgmma.fence tcgen05.alloc tma.bulk tma.multicast"

printf "\n%-18s" "INSTRUCTION"; for a in $ARCHES; do printf "%-10s" "$a"; done; printf "\n"
printf '%s\n' "$(printf '%.0s-' $(seq 1 90))"
for f in $PROBES; do
  printf "%-18s" "$f"
  for a in $ARCHES; do
    sed "s/^\.target .*/.target $a/" "$WORK/$f.ptx" > "$WORK/t.ptx"
    if "$PTXAS" -arch="$a" "$WORK/t.ptx" -o /dev/null 2>/dev/null; then printf "%-10s" "OK"; else printf "%-10s" "--"; fi
  done
  printf "\n"
done

cat <<'NOTES'

READING THIS TABLE
  OK = ptxas ASSEMBLES it for that target. -- = it rejects it.

  Expected result on GB10 (verified with ptxas 12.9):
    mma.sync.*, ldmatrix, cp.async, tma.bulk   OK on sm_121   <- available primitives
    wgmma.fence                                -- on sm_121   <- Hopper only
    tcgen05.alloc                              -- on sm_121   <- SM100 only

  CAVEAT ON tma.multicast: ptxas assembles it for sm_121, but assembling is not
  the same as working. Multicast is only meaningful with thread-block clusters
  larger than 1, and CUTLASS's own SM121 configurations use cluster 1x1x1.
  Sources conflict on whether clusters/DSMEM are genuinely usable on this part.
  Treat "assembles" as necessary but not sufficient, and verify on the device
  before you build anything on it.

  Anything marked -- will fail at BUILD time with a message like:
    "Instruction 'wgmma.fence' not supported on .target 'sm_121'"
  This is the failure mode for GEMM code written for sm_90a (Hopper).
NOTES
