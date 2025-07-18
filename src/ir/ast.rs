use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::{SFold, SMapAccum};

pub use crate::par::LoopPar;

// Reuse the definition of element sizes from the Python AST.
pub use crate::py::ast::ElemSize;
pub use crate::py::ast::UnOp;
pub use crate::py::ast::BinOp;

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Tensor {sz: ElemSize, shape: Vec<i64>},
    Pointer {ty: Box<Type>, count: usize},
    Struct {id: Name},
    Void,
}

impl BinOp {
    pub fn precedence(&self) -> usize {
        match self {
            BinOp::Or => 2,
            BinOp::And => 3,
            BinOp::BitOr => 4,
            BinOp::BitXor => 5,
            BinOp::BitAnd => 6,
            BinOp::Eq | BinOp::Neq => 7,
            BinOp::Leq | BinOp::Geq | BinOp::Lt | BinOp::Gt => 8,
            BinOp::BitShl | BinOp::BitShr => 10,
            BinOp::Add | BinOp::Sub => 11,
            BinOp::Mul | BinOp::FloorDiv | BinOp::Div | BinOp::Rem => 12,
            BinOp::Pow | BinOp::Max | BinOp::Min | BinOp::Atan2 => 20
        }
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    Bool {v: bool, ty: Type, i: Info},
    Int {v: i128, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    IfExpr {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
    StructFieldAccess {target: Box<Expr>, label: String, ty: Type, i: Info},
    TensorAccess {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Call {id: String, args: Vec<Expr>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type},
}

impl Expr {
    pub fn get_type<'a>(&'a self) -> &'a Type {
        match self {
            Expr::Var {ty, ..} => ty,
            Expr::Bool {ty, ..} => ty,
            Expr::Int {ty, ..} => ty,
            Expr::Float {ty, ..} => ty,
            Expr::UnOp {ty, ..} => ty,
            Expr::BinOp {ty, ..} => ty,
            Expr::IfExpr {ty, ..} => ty,
            Expr::StructFieldAccess {ty, ..} => ty,
            Expr::TensorAccess {ty, ..} => ty,
            Expr::Call {ty, ..} => ty,
            Expr::Convert {ty, ..} => ty,
        }
    }
}

impl InfoNode for Expr {
    fn get_info(&self) -> Info {
        match self {
            Expr::Var {i, ..} => i.clone(),
            Expr::Bool {i, ..} => i.clone(),
            Expr::Int {i, ..} => i.clone(),
            Expr::Float {i, ..} => i.clone(),
            Expr::UnOp {i, ..} => i.clone(),
            Expr::BinOp {i, ..} => i.clone(),
            Expr::IfExpr {i, ..} => i.clone(),
            Expr::StructFieldAccess {i, ..} => i.clone(),
            Expr::TensorAccess {i, ..} => i.clone(),
            Expr::Call {i, ..} => i.clone(),
            Expr::Convert {e, ..} => e.get_info(),
        }
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Expr) -> bool {
        match (self, other) {
            (Expr::Var {id: lid, ..}, Expr::Var {id: rid, ..}) => lid.eq(rid),
            (Expr::Bool {v: lv, ..}, Expr::Bool {v: rv, ..}) => lv.eq(rv),
            (Expr::Int {v: lv, ..}, Expr::Int {v: rv, ..}) => lv.eq(rv),
            (Expr::Float {v: lv, ..}, Expr::Float {v: rv, ..}) => lv.eq(rv),
            ( Expr::UnOp {op: lop, arg: larg, ..}
            , Expr::UnOp {op: rop, arg: rarg, ..} ) =>
                lop.eq(rop) && larg.eq(rarg),
            ( Expr::BinOp {lhs: llhs, op: lop, rhs: lrhs, ..}
            , Expr::BinOp {lhs: rlhs, op: rop, rhs: rrhs, ..} ) =>
                llhs.eq(rlhs) && lop.eq(rop) && lrhs.eq(rrhs),
            ( Expr::IfExpr {cond: lcond, thn: lthn, els: lels, ..}
            , Expr::IfExpr {cond: rcond, thn: rthn, els: rels, ..} ) =>
                lcond.eq(rcond) && lthn.eq(rthn) && lels.eq(rels),
            ( Expr::StructFieldAccess {target: ltarget, label: llabel, ..}
            , Expr::StructFieldAccess {target: rtarget, label: rlabel, ..} ) =>
                ltarget.eq(rtarget) && llabel.eq(rlabel),
            ( Expr::TensorAccess {target: ltarget, idx: lidx, ..}
            , Expr::TensorAccess {target: rtarget, idx: ridx, ..} ) =>
                ltarget.eq(rtarget) && lidx.eq(ridx),
            ( Expr::Call {id: lid, args: largs, ..}
            , Expr::Call {id: rid, args: rargs, ..} ) =>
                lid.eq(rid) && largs.eq(rargs),
            (Expr::Convert {e: le, ..}, Expr::Convert {e: re, ..}) => le.eq(re),
            (_, _) => false
        }
    }
}

impl SMapAccum<Expr> for Expr {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Expr) -> Result<(A, Expr), E>
    ) -> Result<(A, Expr), E> {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} => {
                Ok((acc?, self))
            },
            Expr::UnOp {op, arg, ty, i} => {
                let (acc, arg) = f(acc?, *arg)?;
                Ok((acc, Expr::UnOp {op, arg: Box::new(arg), ty, i}))
            },
            Expr::BinOp {lhs, op, rhs, ty, i} => {
                let (acc, lhs) = f(acc?, *lhs)?;
                let (acc, rhs) = f(acc, *rhs)?;
                Ok((acc, Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i}))
            },
            Expr::IfExpr {cond, thn, els, ty, i} => {
                let (acc, cond) = f(acc?, *cond)?;
                let (acc, thn) = f(acc, *thn)?;
                let (acc, els) = f(acc, *els)?;
                Ok((acc, Expr::IfExpr {
                    cond: Box::new(cond), thn: Box::new(thn), els: Box::new(els), ty, i
                }))
            },
            Expr::StructFieldAccess {target, label, ty, i} => {
                let (acc, target) = f(acc?, *target)?;
                Ok((acc, Expr::StructFieldAccess {target: Box::new(target), label, ty, i}))
            },
            Expr::TensorAccess {target, idx, ty, i} => {
                let (acc, target) = f(acc?, *target)?;
                let (acc, idx) = f(acc, *idx)?;
                Ok((acc, Expr::TensorAccess {target: Box::new(target), idx: Box::new(idx), ty, i}))
            },
            Expr::Call {id, args, ty, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::Call {id, args, ty, i}))
            },
            Expr::Convert {e, ty} => {
                let (acc, e) = f(acc?, *e)?;
                Ok((acc, Expr::Convert {e: Box::new(e), ty}))
            },
        }
    }
}

impl SFold<Expr> for Expr {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &Expr) -> Result<A, E>
    ) -> Result<A, E> {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} => acc,
            Expr::UnOp {arg, ..} => f(acc?, arg),
            Expr::BinOp {lhs, rhs, ..} => f(f(acc?, lhs)?, rhs),
            Expr::IfExpr {cond, thn, els, ..} => f(f(f(acc?, cond)?, thn)?, els),
            Expr::StructFieldAccess {target, ..} => f(acc?, target),
            Expr::TensorAccess {target, idx, ..} => f(f(acc?, target)?, idx),
            Expr::Call {args, ..} => args.sfold_result(acc, &f),
            Expr::Convert {e, ..} => f(acc?, e),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Definition {ty: Type, id: Name, expr: Expr, i: Info},
    Assign {dst: Expr, expr: Expr, i: Info},
    SyncPoint {block_local: bool, i: Info},
    For {
        var: Name, lo: Expr, hi: Expr, step: i64, body: Vec<Stmt>,
        par: LoopPar, i: Info
    },
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>, i: Info},
    While {cond: Expr, body: Vec<Stmt>, i: Info},
    Return {value: Expr, i: Info},
    Alloc {id: Name, elem_ty: Type, sz: usize, i: Info},
    Free {id: Name, i: Info},
}

impl InfoNode for Stmt {
    fn get_info(&self) -> Info {
        match self {
            Stmt::Definition {i, ..} => i.clone(),
            Stmt::Assign {i, ..} => i.clone(),
            Stmt::SyncPoint {i, ..} => i.clone(),
            Stmt::For {i, ..} => i.clone(),
            Stmt::If {i, ..} => i.clone(),
            Stmt::While {i, ..} => i.clone(),
            Stmt::Return {i, ..} => i.clone(),
            Stmt::Alloc {i, ..} => i.clone(),
            Stmt::Free {i, ..} => i.clone()
        }
    }
}

impl SMapAccum<Expr> for Stmt {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Expr) -> Result<(A, Expr), E>
    ) -> Result<(A, Stmt), E> {
        match self {
            Stmt::Definition {ty, id, expr, i} => {
                let (acc, expr) = f(acc?, expr)?;
                Ok((acc, Stmt::Definition {ty, id, expr, i}))
            },
            Stmt::Assign {dst, expr, i} => {
                let (acc, dst) = f(acc?, dst)?;
                let (acc, expr) = f(acc, expr)?;
                Ok((acc, Stmt::Assign {dst, expr, i}))
            },
            Stmt::For {var, lo, hi, step, body, par, i} => {
                let (acc, lo) = f(acc?, lo)?;
                let (acc, hi) = f(acc, hi)?;
                Ok((acc, Stmt::For {var, lo, hi, step, body, par, i}))
            },
            Stmt::If {cond, thn, els, i} => {
                let (acc, cond) = f(acc?, cond)?;
                Ok((acc, Stmt::If {cond, thn, els, i}))
            },
            Stmt::While {cond, body, i} => {
                let (acc, cond) = f(acc?, cond)?;
                Ok((acc, Stmt::While {cond, body, i}))
            },
            Stmt::Return {value, i} => {
                let (acc, value) = f(acc?, value)?;
                Ok((acc, Stmt::Return {value, i}))
            },
            Stmt::SyncPoint {..} | Stmt::Alloc {..} | Stmt::Free {..} => {
                Ok((acc?, self))
            },
        }
    }
}

impl SFold<Expr> for Stmt {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &Expr) -> Result<A, E>
    ) -> Result<A, E> {
        match self {
            Stmt::Definition {expr, ..} => f(acc?, expr),
            Stmt::Assign {dst, expr, ..} => f(f(acc?, dst)?, expr),
            Stmt::For {lo, hi, ..} => f(f(acc?, lo)?, hi),
            Stmt::If {cond, ..} => f(acc?, cond),
            Stmt::While {cond, ..} => f(acc?, cond),
            Stmt::Return {value, ..} => f(acc?, value),
            Stmt::SyncPoint {..} | Stmt::Alloc {..} | Stmt::Free {..} => acc
        }
    }
}

impl SMapAccum<Stmt> for Stmt {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Stmt) -> Result<(A, Stmt), E>
    ) -> Result<(A, Stmt), E> {
        match self {
            Stmt::For {var, lo, hi, step, body, par, i} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::For {var, lo, hi, step, body, par, i}))
            },
            Stmt::If {cond, thn, els, i} => {
                let (acc, thn) = thn.smap_accum_l_result(acc, &f)?;
                let (acc, els) = els.smap_accum_l_result(Ok(acc), &f)?;
                Ok((acc, Stmt::If {cond, thn, els, i}))
            },
            Stmt::While {cond, body, i} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::While {cond, body, i}))
            },
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
            Stmt::SyncPoint {..} | Stmt::Alloc {..} | Stmt::Free {..} => {
                Ok((acc?, self))
            }
        }
    }
}

impl SFold<Stmt> for Stmt {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &Stmt) -> Result<A, E>
    ) -> Result<A, E> {
        match self {
            Stmt::For {body, ..} => body.sfold_result(acc, &f),
            Stmt::While {body, ..} => body.sfold_result(acc, &f),
            Stmt::If {thn, els, ..} => els.sfold_result(thn.sfold_result(acc, &f), &f),
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
            Stmt::SyncPoint {..} | Stmt::Alloc {..} | Stmt::Free {..} => acc
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub id: String,
    pub ty: Type,
    pub i: Info
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub id: Name,
    pub ty: Type,
    pub i: Info
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructDef {
    pub id: Name,
    pub fields: Vec<Field>,
    pub i: Info
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunDef {
    pub id: Name,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
    pub res_ty: Type,
    pub i: Info
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ast {
    pub structs: Vec<StructDef>,
    pub defs: Vec<FunDef>,
}
