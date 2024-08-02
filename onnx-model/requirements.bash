
# Constants
RESET='\033[0m'
RED='\033[38;5;1m'
GREEN='\033[38;5;2m'
YELLOW='\033[38;5;3m'
MAGENTA='\033[38;5;5m'
CYAN='\033[38;5;6m'

# check to see if this file is being run or sourced from another script
_is_sourced() {
	# https://unix.stackexchange.com/a/215279
	[ "${#FUNCNAME[@]}" -ge 2 ] \
		&& [ "${FUNCNAME[0]}" = '_is_sourced' ] \
		&& [ "${FUNCNAME[1]}" = 'source' ]
}

_full_cmd_args="$*"
function isTrace(){
    [[ "$_full_cmd_args" = *"trace"* ]] || [[ "$TRACE" =~ true|yes ]]
}
function isDebug(){
    isTrace || [[ "$_full_cmd_args" = *"debug"* ]] || [[ "$DEBUG" =~ true|yes ]]
}
function isError(){
    isDebug || [[ "$_full_cmd_args" = *"error"* ]] || [[ "$ERROR" =~ true|yes ]]
}

function yes(){
    [[ "$_full_cmd_args" =~ *"$1=yes"* ]] || [[ "${!1}" =~ yes ]]
}

function has-option(){
    [[ "$_full_cmd_args" =~ *"-$1"* ]] || [[ "${!1}" =~ yes ]]
}

# 生成一个临时目录 mktemp -u
function current_date(){
  date +'%Y-%m-%d %H:%M:%S%z' 
}

function log {
  local type="$1"
  local msg="$2"

  # printf '%b %s\n' "$type" "$msg" >& 2
  # --rfc-3339=seconds
  # printf '[%s] %b %b\n' "$(date +'%Y-%m-%d %H:%M:%S%z')" "$type" "$msg"
  printf '%b %b\n' "$type" "$msg"
}

function info {
  local msg="$@"

  # echo "$IMENV" | tr 'a-z' 'A-Z'
  [[ -n "$IMENV" ]] && PREFIX="$(echo $IMENV | tr 'a-z' 'A-Z')|"
  log "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S%z'):${PREFIX}INFO]${RESET}" "$msg" >&2
}

function error {
  local msg="$@"
  [[ -n "$IMENV" ]] && PREFIX="$(echo $IMENV | tr 'a-z' 'A-Z')|"
  log "${RED}[$(date +'%Y-%m-%d %H:%M:%S%z'):${PREFIX}ERROR]${RESET}" "$msg" >&2
}

function die {
  error $@
  exit 55
}

function warn {
  local msg="$@"
  [[ -n "$IMENV" ]] && PREFIX="$(echo $IMENV | tr 'a-z' 'A-Z')|"
  log "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S%z'):${PREFIX}WARN]${RESET}" "$msg" >&2
}

function green {
  local a1=$1
  shift
  printf "${GREEN}$a1${RESET}" $*
  echo
}


function fail2die {
  CMD="$@"
  $CMD
  local status=$?
  if [ $status -ne 0 ]; then
    die "Failed($status): $CMD"
  fi
}

# ${varname:?message}
function exit_if_empty() {
  local actual=$1
  local what=$2
  if [[ -z "${actual}" ]]; then
    echo -e "\033[31m[$what]\033[0m required, but it is empty. exit" >&2
    exit -55
  fi
}

function exit_if_empty2() {
  local actual=$1
  if [[ -z "${!actual}" ]]; then
    echo -e "\033[31m[$actual]\033[0m required, but it is empty. exit" >&2
    exit -55
  fi
}

function exit_if_ne() {
  local actual=$1
  exit_if_empty "$actual" "actual is empty"

  local expect=$2
  local what=$3
  if [[ ${actual} -ne ${expect} ]]; then
    echo -e "\033[31m[${actual} -ne ${expect}]\033[0m $what failed. exit" >&2
    exit -1
  else
    echo -e "\033[32m[${actual}]\033[0m $what success"
  fi
}

function exit_if_eq() {
  local actual=$1
  exit_if_empty "$actual" "actual is empty"

  local expect=$2
  local what=$3

  if [[ "x${actual}" == "x${expect}" ]]; then
    echo -e "\033[31m[${actual} -eq ${expect}]\033[0m $what failed. exit" >&2
    exit -1
  fi
}

ERROR=0
function warn_if_ne() {
  local actual=$1
  local expect=$2
  local what=$3
  if [[ ${actual} -ne ${expect} ]]; then
    echo -e "\033[31m[${actual} -ne ${expect}]\033[0m $what failed" >&2
    ERROR=-1
  fi
}

function exit-code() {
  ($@)
  echo $?
}

