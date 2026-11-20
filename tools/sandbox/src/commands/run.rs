// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

mod profile;
mod sample;

use std::boxed::Box;
use std::error;
use std::error::Error;
use std::fs;
use std::io::BufRead;
use std::io::Write;
use std::ops::Bound;
use std::time::Duration;

use octez_riscv::machine_state::memory;
use octez_riscv::machine_state::page_cache;
use octez_riscv::machine_state::page_cache::PageCache;
use octez_riscv::pvm::PvmStatus;
use octez_riscv::pvm::durable_storage::DurableStorageDummy;
use octez_riscv::stepper::StepResult;
use octez_riscv::stepper::Stepper;
use octez_riscv::stepper::StepperStatus;
use octez_riscv::stepper::pvm::PvmStepper;
use octez_riscv_data::mode::Normal;
use tezos_smart_rollup::utils::console::Console;
use tezos_smart_rollup::utils::inbox::InboxBuilder;
use tezos_smart_rollup_encoding::smart_rollup::SmartRollupAddress;

use crate::cli::CommonOptions;
use crate::cli::RunOptions;
use crate::memory_config::MemoryConfigValue;

cfg_if::cfg_if! {
    if #[cfg(feature = "disable-jit")] {
        /// PageCache with interpreted mode selected.
        pub type PageCacheImpl<MC> = page_cache::PageCacheInterpreted<MC>;
    } else if #[cfg(feature = "inline-jit")] {
        /// PageCache with inline jit mode selected.
        pub type PageCacheImpl<MC> = page_cache::PageCacheInlineJit<MC>;
    } else {
        /// PageCache with outline jit mode selected.
        pub type PageCacheImpl<MC> = page_cache::PageCacheOutlineJit<MC>;
    }
}

pub fn run_with_memory_config<MC: memory::MemoryConfig>(
    opts: RunOptions,
) -> Result<(), Box<dyn Error>> {
    let program = fs::read(&opts.input)?;

    let mut stepper = make_pvm_stepper::<MC, PageCacheImpl<MC>>(program.as_slice(), &opts.common)?;

    // No inbox file: each stdin line is one message, and its outbox
    // output is written before the next line is read. An inbox file keeps
    // the batch behaviour of loading every message first.
    let steps = if opts.sample_interval_us.is_none() && opts.common.inbox.file.is_none() {
        run_stdin(&mut stepper, opts.common.max_steps)?
    } else {
        match opts.sample_interval_us {
            None => {
                let steps = run_stepper(&mut stepper, opts.common.max_steps)?;
                print_outbox(&stepper)?;
                steps
            }
            Some(sample_interval_us) => {
                let sample_interval = Duration::from_micros(sample_interval_us);
                profile::profile_stepper(
                    stepper,
                    program.as_slice(),
                    sample_interval,
                    opts.common.max_steps,
                    opts.output.as_ref(),
                )?
            }
        }
    };

    if opts.print_steps {
        println!("Run consumed {steps} steps.");
    }

    Ok(())
}

pub fn run(opts: RunOptions) -> Result<(), Box<dyn Error>> {
    // Promote the memory configuration value to the appropriate type, then continue.
    match opts.common.memory_config {
        MemoryConfigValue::M64M => run_with_memory_config::<memory::M64M>(opts),
        MemoryConfigValue::M1G => run_with_memory_config::<memory::M1G>(opts),
        MemoryConfigValue::M4G => run_with_memory_config::<memory::M4G>(opts),
        MemoryConfigValue::M16G => run_with_memory_config::<memory::M16G>(opts),
        MemoryConfigValue::M64G => run_with_memory_config::<memory::M64G>(opts),
    }
}

type PvmStepperRunner<MC, PC> = PvmStepper<Console<'static>, MC, DurableStorageDummy, PC, Normal>;

pub(crate) fn make_pvm_stepper<MC: memory::MemoryConfig, PC: PageCache<MC, Normal>>(
    program: &[u8],
    common: &CommonOptions,
) -> Result<PvmStepperRunner<MC, PC>, Box<dyn error::Error>> {
    let mut inbox = InboxBuilder::new();
    if let Some(inbox_file) = &common.inbox.file {
        inbox.load_from_file(inbox_file)?;
    }

    let rollup_address = SmartRollupAddress::from_b58check(common.inbox.address.as_str())?;

    let console = if common.timings {
        Console::with_timings()
    } else {
        Console::new()
    };

    let stepper = PvmStepper::<_, MC, DurableStorageDummy, PC, Normal>::new(
        program,
        inbox.build(),
        console,
        rollup_address.into_hash().as_ref().try_into()?,
        common.inbox.origination_level,
        common.preimage.preimages_dir.clone(),
    )?;

    Ok(stepper)
}

const MESSAGE_MAX: usize = 4096;

/// Run until the guest asks for a message or leaves the evaluation loop.
fn step_until_input<MC, PC>(
    stepper: &mut PvmStepperRunner<MC, PC>,
    mut step_bounds: Bound<usize>,
) -> Result<(usize, bool), Box<dyn Error>>
where
    MC: memory::MemoryConfig,
    PC: PageCache<MC, Normal>,
{
    let mut total = 0usize;
    loop {
        if steps_exhausted(step_bounds) {
            return Err(format!("{:?}", StepperStatus::Running { steps: total }).into());
        }
        match stepper.step_max_once(step_bounds, true) {
            StepperStatus::Running { steps } => {
                total = total.saturating_add(steps);
                step_bounds = bound_saturating_sub(step_bounds, steps);
                if stepper.pvm_status() == PvmStatus::WaitingForInput {
                    return Ok((total, true));
                }
                if steps < 1 {
                    return Err(format!("{:?}", StepperStatus::Running { steps: total }).into());
                }
            }
            StepperStatus::Exited {
                steps,
                success: true,
                ..
            } => return Ok((total.saturating_add(steps), false)),
            status => return Err(format!("{status:?}").into()),
        }
    }
}

fn steps_exhausted(bound: Bound<usize>) -> bool {
    matches!(
        bound,
        Bound::Included(0) | Bound::Excluded(0) | Bound::Excluded(1)
    )
}

fn bound_saturating_sub(bound: Bound<usize>, steps: usize) -> Bound<usize> {
    match bound {
        Bound::Included(n) => Bound::Included(n.saturating_sub(steps)),
        Bound::Excluded(n) => Bound::Excluded(n.saturating_sub(steps)),
        Bound::Unbounded => Bound::Unbounded,
    }
}

/// One stdin line becomes one external inbox message.
///
/// A JSON string is the program text. Any other JSON value is delivered as the line itself.
fn message_from_line(line: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let trimmed = line.trim();
    let value: serde_json::Value =
        serde_json::from_str(trimmed).map_err(|err| format!("stdin: {err}"))?;
    let text = match value {
        serde_json::Value::String(program) => program,
        _ => trimmed.to_string(),
    };
    if text.len() + 1 > MESSAGE_MAX {
        return Err("stdin message exceeds 4095 bytes".into());
    }
    let mut payload = Vec::with_capacity(text.len() + 1);
    payload.push(1);
    payload.extend(text.into_bytes());
    Ok(payload)
}

fn read_message(input: &mut impl BufRead) -> Result<Option<Vec<u8>>, Box<dyn Error>> {
    let mut line = String::new();
    loop {
        line.clear();
        if input.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        if line.trim().is_empty() {
            continue;
        }
        return Ok(Some(message_from_line(&line)?));
    }
}

fn take_new_outbox<MC, PC>(
    stepper: &PvmStepperRunner<MC, PC>,
    printed: &mut Option<(u32, u32)>,
) -> Vec<u8>
where
    MC: memory::MemoryConfig,
    PC: PageCache<MC, Normal>,
{
    let mut bytes = Vec::new();
    for (level, messages) in stepper.outbox_messages() {
        for (index, message) in messages.into_iter().enumerate() {
            let index = index as u32;
            if printed.is_some_and(|done| (level, index) <= done) {
                continue;
            }
            bytes.extend_from_slice(&message);
            *printed = Some((level, index));
        }
    }
    bytes
}

fn write_new_outbox<MC, PC>(
    stepper: &PvmStepperRunner<MC, PC>,
    stdout: &mut impl Write,
    printed: &mut Option<(u32, u32)>,
) -> Result<(), Box<dyn Error>>
where
    MC: memory::MemoryConfig,
    PC: PageCache<MC, Normal>,
{
    stdout.write_all(&take_new_outbox(stepper, printed))?;
    stdout.flush()?;
    Ok(())
}

/// One JSON line per inbox message so a host kernel can detect cell completion.
fn write_framed_outbox<MC, PC>(
    stepper: &PvmStepperRunner<MC, PC>,
    stdout: &mut impl Write,
    printed: &mut Option<(u32, u32)>,
) -> Result<(), Box<dyn Error>>
where
    MC: memory::MemoryConfig,
    PC: PageCache<MC, Normal>,
{
    let bytes = take_new_outbox(stepper, printed);
    let output = String::from_utf8_lossy(&bytes).into_owned();
    serde_json::to_writer(&mut *stdout, &serde_json::json!({ "output": output }))?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn run_stdin<MC, PC>(
    stepper: &mut PvmStepperRunner<MC, PC>,
    max_steps: Option<usize>,
) -> Result<usize, Box<dyn Error>>
where
    MC: memory::MemoryConfig,
    PC: PageCache<MC, Normal>,
{
    let mut step_bounds = match max_steps {
        Some(max_steps) => Bound::Included(max_steps),
        None => Bound::Unbounded,
    };
    let mut total = 0usize;
    let mut level = 0u32;
    let mut printed = None;
    let mut expect_reply = false;
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut stdout = std::io::stdout();

    loop {
        let (steps, waiting) = step_until_input(stepper, step_bounds)?;
        total = total.saturating_add(steps);
        step_bounds = bound_saturating_sub(step_bounds, steps);
        if expect_reply {
            write_framed_outbox(stepper, &mut stdout, &mut printed)?;
            expect_reply = false;
        }
        if !waiting {
            return Ok(total);
        }
        let Some(payload) = read_message(&mut input)? else {
            return Ok(total);
        };
        if !stepper.provide_inbox_message(level, 0, &payload) {
            return Err("guest was not waiting for an inbox message".into());
        }
        level = level.saturating_add(1);
        expect_reply = true;
    }
}

fn print_outbox<MC, PC>(stepper: &PvmStepperRunner<MC, PC>) -> Result<(), Box<dyn Error>>
where
    MC: memory::MemoryConfig,
    PC: PageCache<MC, Normal>,
{
    let mut stdout = std::io::stdout().lock();
    for (_level, messages) in stepper.outbox_messages() {
        for message in messages {
            stdout.write_all(&message)?;
        }
    }
    stdout.flush()?;
    Ok(())
}

fn run_stepper(
    stepper: &mut impl Stepper,
    max_steps: Option<usize>,
) -> Result<usize, Box<dyn Error>> {
    let max_steps = match max_steps {
        Some(max_steps) => Bound::Included(max_steps),
        None => Bound::Unbounded,
    };

    let result = stepper.step_max(max_steps);

    match result.to_stepper_status() {
        StepperStatus::Exited {
            success: true,
            steps,
            ..
        } => Ok(steps),
        result => Err(format!("{result:?}").into()),
    }
}
