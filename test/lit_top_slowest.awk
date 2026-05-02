# Post-filter for the lit output piped from `make test`.
#
# - Drops the per-bucket "Tests Times:" histogram (noise).
# - Trims the "Slowest Tests:" section to the top 5 entries while
#   preserving its header / divider lines.
# - Otherwise passes the stream through unchanged so the per-test
#   PASS/FAIL lines and the final pass/fail counters still print.

BEGIN {
  in_slow = 0      # 1 while inside the "Slowest Tests:" block
  slow_kept = 0    # how many timed entries we have already printed
  in_hist = 0      # 1 while inside the "Tests Times:" histogram block
}

# Detect the start of the slowest-tests block.
/^Slowest Tests:[[:space:]]*$/ {
  in_slow = 1
  slow_kept = 0
  print "Slowest 5 Tests:"
  next
}

# Detect the start of the bucket histogram block; skip the whole block.
/^Tests Times:[[:space:]]*$/ {
  in_hist = 1
  next
}

# Inside the histogram block: skip until a blank line ends the block.
in_hist {
  if ($0 ~ /^[[:space:]]*$/) {
    in_hist = 0
    print
  }
  next
}

# Inside the slowest-tests block.
in_slow {
  # The block opens with "----" divider, then `<sec>: <test>` lines, then
  # closes with another "----" divider followed by blank line.
  if ($0 ~ /^-+$/) {
    # Always print dividers (open and close).
    print
    if (slow_kept >= 5) {
      # We have already trimmed; the closing divider takes us out of
      # the block.
      in_slow = 0
    }
    next
  }
  if ($0 ~ /^[0-9]+\.[0-9]+s:[[:space:]]/) {
    if (slow_kept < 5) {
      print
      slow_kept++
    }
    # Drop entries past 5; keep scanning for the closing divider.
    next
  }
  # Blank line or anything else inside the block: pass through and exit.
  print
  if ($0 ~ /^[[:space:]]*$/)
    in_slow = 0
  next
}

{ print }
