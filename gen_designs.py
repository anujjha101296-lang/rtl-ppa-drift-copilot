#!/usr/bin/env python3
"""Generate a corpus of parameterized synthesizable RTL designs.

Every design has top module `top`. Families span datapath (adders, ALUs,
multipliers, shifters), control (FSMs, counters, priority logic), and
random combinational DAGs so the model sees structural diversity.
Output: designs/<family>__<params>.v  +  designs/manifest.json
"""
import json, os, random, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "designs")
os.makedirs(OUT, exist_ok=True)
designs = []  # (family, params dict, verilog)


def add(family, params, v):
    designs.append((family, params, v))


# ---------------------------------------------------------------- datapath
def rca(W):
    return f"""module top(input [{W-1}:0] a, b, input cin, output [{W-1}:0] s, output cout);
  wire [{W}:0] c; assign c[0]=cin;
  genvar i;
  generate for (i=0;i<{W};i=i+1) begin: fa
    assign s[i]=a[i]^b[i]^c[i];
    assign c[i+1]=(a[i]&b[i])|(c[i]&(a[i]^b[i]));
  end endgenerate
  assign cout=c[{W}];
endmodule"""

def add_behav(W):
    return f"""module top(input [{W-1}:0] a,b, output [{W}:0] s);
  assign s = a + b;
endmodule"""

def addsub(W):
    return f"""module top(input [{W-1}:0] a,b, input sub, output [{W}:0] s);
  assign s = sub ? ({{1'b0,a}} - {{1'b0,b}}) : ({{1'b0,a}} + {{1'b0,b}});
endmodule"""

def cla_group(W, G):
    # carry-lookahead with group size G (explicit p/g)
    n = W
    return f"""module top(input [{n-1}:0] a,b, input cin, output [{n-1}:0] s, output cout);
  wire [{n-1}:0] p = a ^ b, g = a & b;
  wire [{n}:0] c; assign c[0] = cin;
  genvar i;
  generate for (i=0;i<{n};i=i+1) begin: cc
    if (i % {G} == 0) begin : grp
      // lookahead across group
      wire [{G}:0] cl;
      assign cl[0] = c[i];
      genvar j;
      for (j=0;j<{G};j=j+1) begin : l
        if (i+j < {n}) assign cl[j+1] = g[i+j] | (p[i+j] & cl[j]);
        else assign cl[j+1] = 1'b0;
      end
      for (j=0;j<{G};j=j+1) begin : w
        if (i+j < {n}) assign c[i+j+1] = cl[j+1];
      end
    end
  end endgenerate
  assign s = p ^ c[{n-1}:0];
  assign cout = c[{n}];
endmodule"""

def mult(W):
    return f"""module top(input [{W-1}:0] a,b, output [{2*W-1}:0] p);
  assign p = a * b;
endmodule"""

def mult_seq(W):
    # shift-add multiplier, registered result
    return f"""module top(input clk, rst, input [{W-1}:0] a,b, output reg [{2*W-1}:0] p);
  reg [{2*W-1}:0] acc; reg [{W-1}:0] m; reg [{2*W-1}:0] ax;
  always @(posedge clk) begin
    if (rst) begin acc<=0; m<=0; ax<=0; p<=0; end
    else begin
      ax <= {{{W}'b0,a}}; m <= b;
      acc <= (m[0] ? ax : {2*W}'b0) + (acc >> 1);
      p <= acc;
    end
  end
endmodule"""

def alu(W, nops):
    ops = ["a+b", "a-b", "a&b", "a|b", "a^b", "~(a&b)", "a<<1", "a>>1",
           "{{{0}{{a<b}}}}", "a^(a>>1)", "b-a", "~a"]
    case_lines = []
    for i in range(nops):
        if i == 8:
            expr = "{{" + str(W - 1) + "{1'b0}}, (a<b)}"
        else:
            expr = ops[i].format(W - 1)
        case_lines.append(f"      3'd{i}: y = {expr};")
    cases = "\\n".join(case_lines)
    return f"""module top(input [{W-1}:0] a,b, input [2:0] op, output reg [{W-1}:0] y);
  always @* begin
    case (op)
{cases}
      default: y = {W}'b0;
    endcase
  end
endmodule"""

def barrel(W):
    L = max(1, (W-1).bit_length())
    return f"""module top(input [{W-1}:0] d, input [{L-1}:0] sh, output [{W-1}:0] y);
  assign y = d << sh;
endmodule"""

def rotate(W):
    L = max(1, (W-1).bit_length())
    return f"""module top(input [{W-1}:0] d, input [{L-1}:0] sh, output [{W-1}:0] y);
  wire [{2*W-1}:0] dd = {{d,d}};
  wire [{2*W-1}:0] r = dd >> ({W}-sh);
  assign y = r[{W-1}:0];
endmodule"""

def popcount(W):
    return f"""module top(input [{W-1}:0] d, output reg [{max(1,W.bit_length())-1}:0] n);
  integer i;
  always @* begin n = 0; for (i=0;i<{W};i=i+1) n = n + d[i]; end
endmodule"""

# ------------------------------------------------------------- comparison
def cmp_chain(W):
    return f"""module top(input [{W-1}:0] a,b, output eq, output lt);
  wire [{W}:0] e, l; assign e[{W}]=1'b1; assign l[{W}]=1'b0;
  genvar i;
  generate for (i={W-1};i>=0;i=i-1) begin: c
    assign l[i] = l[i+1] | (e[i+1] & ~a[i] & b[i]);
    assign e[i] = e[i+1] & ~(a[i]^b[i]);
  end endgenerate
  assign eq = e[0]; assign lt = l[0];
endmodule"""

def cmp_behav(W):
    return f"""module top(input [{W-1}:0] a,b, output eq, lt, gt);
  assign eq = (a==b); assign lt = (a<b); assign gt = (a>b);
endmodule"""

def parity_tree(W):
    return f"""module top(input [{W-1}:0] d, output p); assign p = ^d; endmodule"""

def parity_chain(W):
    return f"""module top(input [{W-1}:0] d, output p);
  wire [{W}:0] x; assign x[0]=1'b0;
  genvar i; generate for(i=0;i<{W};i=i+1) begin:g assign x[i+1]=x[i]^d[i]; end endgenerate
  assign p = x[{W}];
endmodule"""

def and_or_chain(W):
    return f"""module top(input [{W-1}:0] d, output y);
  wire [{W}:0] t; assign t[0]=d[0];
  genvar i; generate for(i=1;i<{W};i=i+1) begin:g
    if (i%2) assign t[i]=t[i-1]&d[i]; else assign t[i]=t[i-1]|d[i];
  end endgenerate
  assign y=t[{W-1}];
endmodule"""

# ---------------------------------------------------------------- muxing
def mux_tree(N, W):
    L = max(1, (N-1).bit_length())
    ports = ", ".join(f"input [{W-1}:0] d{i}" for i in range(N))
    cases = "\n".join(f"      {L}'d{i}: y = d{i};" for i in range(N))
    return f"""module top({ports}, input [{L-1}:0] s, output reg [{W-1}:0] y);
  always @* begin
    case (s)
{cases}
      default: y = {W}'b0;
    endcase
  end
endmodule"""

def mux_chain(N, W):
    ports = ", ".join(f"input [{W-1}:0] d{i}" for i in range(N))
    expr = f"d{N-1}"
    for i in range(N-2, -1, -1):
        expr = f"(s=={i}) ? d{i} : ({expr})"
    L = max(1, (N-1).bit_length())
    return f"""module top({ports}, input [{L-1}:0] s, output [{W-1}:0] y);
  assign y = {expr};
endmodule"""

def prio_enc(N):
    L = max(1, (N-1).bit_length())
    cases = "\n".join(f"    else if (d[{i}]) begin y = {L}'d{i}; v = 1'b1; end" for i in range(1, N))
    return f"""module top(input [{N-1}:0] d, output reg [{L-1}:0] y, output reg v);
  always @* begin
    y = 0; v = 1'b0;
    if (d[0]) begin y = 0; v = 1'b1; end
{cases}
  end
endmodule"""

def decoder(L):
    return f"""module top(input [{L-1}:0] a, input en, output [{2**L-1}:0] y);
  assign y = en ? ({2**L}'b1 << a) : {2**L}'b0;
endmodule"""

# ------------------------------------------------------------ sequential
def shreg(W, taps):
    t = " ^ ".join(f"q[{i}]" for i in taps)
    return f"""module top(input clk, rst, din, output [{W-1}:0] q_o, output fb);
  reg [{W-1}:0] q;
  assign fb = {t};
  always @(posedge clk) if (rst) q <= 0; else q <= {{q[{W-2}:0], din ^ fb}};
  assign q_o = q;
endmodule"""

def counter(W, mode):
    if mode == "up":
        body = "cnt <= cnt + 1;"
    elif mode == "updown":
        body = "cnt <= dn ? cnt - 1 : cnt + 1;"
    else:  # load/cmp
        body = f"cnt <= (cnt == {W}'d{2**W-3}) ? {W}'d0 : cnt + 1;"
    return f"""module top(input clk, rst, en, dn, output reg [{W-1}:0] cnt, output hit);
  always @(posedge clk) if (rst) cnt <= 0; else if (en) begin {body} end
  assign hit = &cnt;
endmodule"""

def lfsr(W, taps):
    t = " ^ ".join(f"r[{i}]" for i in taps)
    return f"""module top(input clk, rst, output [{W-1}:0] o);
  reg [{W-1}:0] r;
  always @(posedge clk) if (rst) r <= {W}'d1; else r <= {{r[{W-2}:0], {t}}};
  assign o = r;
endmodule"""

def crc(W, poly):
    return f"""module top(input clk, rst, din, output reg [{W-1}:0] c);
  wire fb = c[{W-1}] ^ din;
  always @(posedge clk) if (rst) c <= 0;
    else c <= {{c[{W-2}:0],1'b0}} ^ ({W}'h{poly:x} & {{{W}{{fb}}}});
endmodule"""

def fsm(S, I, enc, seed):
    rnd = random.Random(seed)
    sb = max(1, (S-1).bit_length())
    cases = []
    for s in range(S):
        arms = []
        for k in range(1 << I):
            arms.append(f"        {I}'d{k}: ns = {sb}'d{rnd.randrange(S)};")
        out = rnd.randrange(2)
        cases.append(f"""      {sb}'d{s}: begin
        case (in)
{chr(10).join(arms)}
        endcase
        o = 1'b{out};
      end""")
    return f"""module top(input clk, rst, input [{I-1}:0] in, output reg o);
  (* fsm_encoding = "{enc}" *)
  reg [{sb-1}:0] st, ns;
  always @* begin
    ns = st; o = 1'b0;
    case (st)
{chr(10).join(cases)}
      default: ns = {sb}'d0;
    endcase
  end
  always @(posedge clk) if (rst) st <= 0; else st <= ns;
endmodule"""

def pipe_add(W, stages):
    # pipelined accumulate: tree of registered adds
    regs = "\n".join(f"  reg [{W-1}:0] s{i};" for i in range(stages))
    body = "\n".join(
        f"    s{i} <= {'a' if i==0 else f's{i-1}'} + {'b' if i==0 else f's{i-1}'};" for i in range(stages))
    return f"""module top(input clk, input [{W-1}:0] a,b, output [{W-1}:0] y);
{regs}
  always @(posedge clk) begin
{body}
  end
  assign y = s{stages-1};
endmodule"""

def mac(W):
    return f"""module top(input clk, rst, input [{W-1}:0] a,b, output reg [{2*W-1}:0] acc);
  always @(posedge clk) if (rst) acc <= 0; else acc <= acc + a*b;
endmodule"""

# ------------------------------------------------------- random DAG logic
def rand_dag(nin, ngate, seed, mix):
    rnd = random.Random(seed)
    ops = {"and": "&", "or": "|", "xor": "^"}
    weights = {"and": mix[0], "or": mix[1], "xor": mix[2]}
    names = [f"d[{i}]" for i in range(nin)]
    lines = []
    for g in range(ngate):
        k = rnd.choices(list(ops), weights=[weights[o] for o in ops])[0]
        # bias toward recent nodes => deeper graphs when locality high
        loc = min(len(names), 8)
        a = rnd.choice(names[-loc:]) if rnd.random() < 0.6 else rnd.choice(names)
        b = rnd.choice(names)
        inv = "~" if rnd.random() < 0.3 else ""
        lines.append(f"  wire n{g} = {inv}({a} {ops[k]} {b});")
        names.append(f"n{g}")
    outs = names[-max(2, ngate // 8):]
    ow = len(outs)
    return f"""module top(input [{nin-1}:0] d, output [{ow-1}:0] y);
{chr(10).join(lines)}
  assign y = {{{', '.join(outs)}}};
endmodule"""

# ------------------------------------------------------------- build sweep
for W in (4, 6, 8, 12, 16, 24, 32, 48, 64):
    add("rca", {"W": W}, rca(W))
    add("add_behav", {"W": W}, add_behav(W))
    add("addsub", {"W": W}, addsub(W))
    add("cmp_chain", {"W": W}, cmp_chain(W))
    add("cmp_behav", {"W": W}, cmp_behav(W))
    add("parity_tree", {"W": W}, parity_tree(W))
    add("parity_chain", {"W": W}, parity_chain(W))
    add("and_or_chain", {"W": W}, and_or_chain(W))
    add("barrel", {"W": W}, barrel(W))
    add("rotate", {"W": W}, rotate(W))
    add("popcount", {"W": W}, popcount(W))
    add("counter_up", {"W": W}, counter(W, "up"))
    add("counter_ud", {"W": W}, counter(W, "updown"))
    add("counter_wrap", {"W": W}, counter(W, "wrap"))
for W in (8, 12, 16, 24, 32):
    for G in (2, 4, 8):
        add("cla", {"W": W, "G": G}, cla_group(W, G))
for W in (4, 6, 8, 10, 12, 16):
    add("mult", {"W": W}, mult(W))
    add("mult_seq", {"W": W}, mult_seq(W))
    add("mac", {"W": W}, mac(W))
for W in (8, 16, 32):
    for nops in (4, 6, 8):
        add("alu", {"W": W, "nops": nops}, alu(W, nops))
for N in (4, 8, 16, 32):
    for W in (1, 4, 8):
        add("mux_tree", {"N": N, "W": W}, mux_tree(N, W))
        add("mux_chain", {"N": N, "W": W}, mux_chain(N, W))
for N in (4, 8, 16, 32, 64):
    add("prio_enc", {"N": N}, prio_enc(N))
for L in (2, 3, 4, 5, 6):
    add("decoder", {"L": L}, decoder(L))
for W, taps in ((8, (7, 5, 4, 3)), (16, (15, 14, 12, 3)), (24, (23, 22, 21, 16)),
                (32, (31, 21, 1, 0)), (12, (11, 10, 9, 3)), (10, (9, 6))):
    add("shreg", {"W": W, "ntaps": len(taps)}, shreg(W, taps))
    add("lfsr", {"W": W, "ntaps": len(taps)}, lfsr(W, taps))
for W, poly in ((8, 0x07), (16, 0x1021), (16, 0x8005), (32, 0x04C11DB7), (12, 0x80F), (24, 0x864CFB)):
    add("crc", {"W": W, "poly": poly}, crc(W, poly))
for S in (4, 6, 8, 12, 16, 24):
    for I in (1, 2, 3):
        for enc in ("binary", "one-hot"):
            add("fsm", {"S": S, "I": I, "enc": enc}, fsm(S, I, enc, seed=S * 100 + I * 7 + len(enc)))
for W in (8, 16, 32):
    for st in (1, 2, 4):
        add("pipe_add", {"W": W, "stages": st}, pipe_add(W, st))
rr = random.Random(2024)
for k in range(50):
    nin = rr.choice((8, 12, 16, 24, 32))
    ng = rr.choice((20, 40, 80, 160, 320))
    mix = (rr.randint(1, 5), rr.randint(1, 5), rr.randint(0, 4))
    add("rand_dag", {"nin": nin, "ngate": ng, "seed": 9000 + k, "mix": list(mix)},
        rand_dag(nin, ng, 9000 + k, mix))

manifest = []
for i, (fam, params, v) in enumerate(designs):
    ptag = "_".join(str(x) for x in params.values()).replace("[", "").replace("]", "").replace(",", "-").replace(" ", "")
    name = f"{i:04d}_{fam}__{ptag}"
    with open(os.path.join(OUT, name + ".v"), "w") as f:
        f.write(v + "\n")
    manifest.append({"name": name, "family": fam, "params": params})
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print(f"generated {len(manifest)} designs in {len(set(m['family'] for m in manifest))} families")
