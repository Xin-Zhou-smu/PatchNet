(*
 * This file is part of PatchNet, licensed under the terms of the GPL v2.
 * See copyright.txt in the PatchNet source code for more information.
 * The PatchNet source code can be obtained at
 * https://github.com/hvdthong/PatchNetTool
 *)

let linux = ref "/dev/shm/linux"
let before_linux = "/dev/shm/linux-before"
(*let after_linux = "/mnt/ramdisk/linux-after"*)
let after_linux = "/dev/shm/linux-after"
let stable = ref "/dev/shm/linux-stable"

let process_output_to_list2 = fun command ->
  let chan = Unix.open_process_in command in
  let res = ref ([] : string list) in
  let rec process_otl_aux () =
    let e = input_line chan in
    res := e::!res;
    process_otl_aux() in
  try process_otl_aux ()
  with End_of_file ->
    let stat = Unix.close_process_in chan in (List.rev !res,stat)
let cmd_to_list command =
  let (l,_) = process_output_to_list2 command in l
let process_output_to_list = cmd_to_list
let cmd_to_list_and_status = process_output_to_list2

let safe_int_of_string key s =
  try int_of_string s
  with x -> failwith (Printf.sprintf "%d: ios failure on %s\n" key s)

(* from later ocaml *)
let is_space = function
  |  ' ' | '\012' | '\n' | '\r' | '\t' -> true
  |  _ -> false

let trim s =
  let len = String.length s in
  let i = ref 0 in
  while !i < len && try is_space (String.get s !i) with _ -> false do
    incr i
  done;
  let j = ref (len - 1) in
  while !j >= !i && try is_space (String.get s !j) with _ -> false do
    decr j
  done;
  if !i = 0 && !j = len - 1 then
    s
  else if !j >= !i then
    String.sub s !i (!j - !i + 1)
  else
    ""
let cores = ref 4
let chunksize = 1
let tmpdir = ref "/tmp"

let union l1 l2 =
  List.fold_left (fun prev x -> if List.mem x prev then prev else x::prev)
    l2 l1

type ('a,'b) either = Left of 'a | Right of 'b

let hashadd tbl k v =
  let cell =
    try Hashtbl.find tbl k
    with Not_found ->
      let cell = ref [] in
      Hashtbl.add tbl k cell;
      cell in
  if not (List.mem v !cell) then cell := v :: !cell

(* thanks to Francois Berenger *)
let rec takedrop n l =
  let rec loop n acc = function
      [] -> (List.rev acc,[])
    | l1 when n = 0 -> (List.rev acc,l1)
    | x::xs -> loop (n-1) (x::acc) xs in
  loop n [] l

let parfold_compat
    ?(init = fun (_rank: int) -> ()) ?(finalize = fun () -> ())
    ?(ncores: int option) ?(chunksize: int option) (f: 'a -> 'b -> 'b)
    (l: 'a list) (init_acc: 'b) (acc_fun: 'b -> 'b -> 'b): 'b =
  flush stdout; flush stderr;
  let nprocs = match ncores with
  | None -> 1 (* if the user doesn't know the number of cores to use,
                 we don't know better *)
  | Some x -> x in
  let csize = match chunksize with
  | None -> 1
  | Some x -> x in
  if nprocs <= 1 then
    List.fold_left (fun acc x -> f x acc) init_acc l
  else
    let input = ref l in
    let demux () = match !input with
    | [] -> raise Parany.End_of_input
    | _ ->
        let this_chunk, rest = takedrop csize !input in
        input := rest;
        this_chunk in
    let work xs =
      List.fold_left (fun acc x -> f x acc) init_acc xs in
    let output = ref init_acc in
    let mux x =
      output := acc_fun !output x in
      (* parallel work *)
    Parany.run ~init ~finalize
        (* leave csize=1 bellow *)
      ~preserve:false ~core_pin:true ~csize:1 nprocs ~demux ~work ~mux;
    !output
