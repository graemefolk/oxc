//! Dynamic AST node storage used by [`SemanticBuilder`] during traversal.
//!
//! Two backends are available:
//!
//! 1. [`NodeStorage::Full`] — every node is recorded in [`AstNodes`], giving
//!    random access by [`NodeId`] after the build finishes. Required by
//!    consumers that walk the whole tree (linter, formatter, mangler).
//! 2. [`NodeStorage::Ancestors`] — only the *live ancestor chain*
//!    (`root..=current`) is retained, via [`AncestorStack`]. This is enough for
//!    the binder, the class-table builder, and the syntax checker — which only
//!    ever look *upwards* from the current node — while avoiding the per-node
//!    allocations of full storage. Used by pipelines that discard the AST nodes
//!    and keep only [`Scoping`] (transform, minify, define/inject).
//!
//! Both backends allocate [`NodeId`]s from the same monotonic counter, so the
//! ids stored in [`Scoping`] are identical regardless of the backend.
//!
//! [`SemanticBuilder`]: crate::SemanticBuilder
//! [`Scoping`]: crate::Scoping

use itertools::Either;

use oxc_ast::AstKind;
use oxc_data_structures::stack::Stack;
use oxc_syntax::{
    node::{NodeFlags, NodeId},
    scope::ScopeId,
};

#[cfg(feature = "cfg")]
use oxc_cfg::BlockNodeId;

use super::AstNode;
use crate::node::AstNodes;

/// A single entry in the [`AncestorStack`].
struct StackEntry<'a> {
    id: NodeId,
    node: AstNode<'a>,
    flags: NodeFlags,
}

/// Stores only the live ancestor chain (`root..=current`) during traversal.
///
/// Pushed on `enter_node`, popped on `leave_node`, so at any point the stack
/// holds the path from the root [`Program`] down to the node currently being
/// visited. This serves every *upward* query the builder needs (parents and
/// ancestors) without retaining the entire tree.
///
/// [`Program`]: oxc_ast::ast::Program
#[derive(Default)]
pub struct AncestorStack<'a> {
    /// `stack[0]` is the root, `stack.last()` is the current node.
    ///
    /// Uses the cursor-based [`Stack`] (rather than `Vec`) for fast push / pop /
    /// `last`, which run on every node entered and exited.
    stack: Stack<StackEntry<'a>>,
    /// Total number of nodes created. Doubles as the allocator for the next
    /// [`NodeId`], keeping ids consistent with full storage.
    len: u32,
}

impl<'a> AncestorStack<'a> {
    /// Find the stack position of a live ancestor `id`.
    ///
    /// The current node (top of stack) is the overwhelmingly common case and is
    /// checked first. Other ancestors (e.g. a scope's or class's node) require a
    /// scan, but the stack depth equals the AST nesting depth, which is small.
    #[inline]
    fn position(&self, id: NodeId) -> usize {
        if let Some(last) = self.stack.last()
            && last.id == id
        {
            return self.stack.len() - 1;
        }
        self.stack.iter().rposition(|entry| entry.id == id).expect(
            "`NodeId` is not a live ancestor (not available in parent-pointer storage mode)",
        )
    }

    #[inline]
    fn next_id(&mut self) -> NodeId {
        let id = NodeId::new(self.len as usize);
        self.len += 1;
        id
    }

    fn add_node(&mut self, kind: AstKind<'a>, scope_id: ScopeId, flags: NodeFlags) -> NodeId {
        let node_id = self.next_id();
        kind.set_node_id(node_id);
        self.stack.push(StackEntry { id: node_id, node: AstNode::new(kind, scope_id), flags });
        node_id
    }

    fn add_program_node(
        &mut self,
        kind: AstKind<'a>,
        scope_id: ScopeId,
        flags: NodeFlags,
    ) -> NodeId {
        debug_assert!(self.stack.is_empty(), "Program node must be the first node in the AST.");
        let node_id = self.next_id();
        debug_assert_eq!(node_id, NodeId::ROOT);
        kind.set_node_id(node_id);
        self.stack.push(StackEntry { id: node_id, node: AstNode::new(kind, scope_id), flags });
        node_id
    }

    /// Pop the current node and return its parent's id.
    fn pop_node(&mut self) -> NodeId {
        self.stack.pop();
        self.stack.last().map_or(NodeId::ROOT, |entry| entry.id)
    }

    #[inline]
    fn get_node(&self, id: NodeId) -> &AstNode<'a> {
        &self.stack[self.position(id)].node
    }

    #[inline]
    fn parent_id(&self, id: NodeId) -> NodeId {
        let pos = self.position(id);
        if pos == 0 { NodeId::ROOT } else { self.stack[pos - 1].id }
    }

    #[inline]
    fn flags_mut(&mut self, id: NodeId) -> &mut NodeFlags {
        let pos = self.position(id);
        &mut self.stack[pos].flags
    }

    #[inline]
    fn ancestor_ids(&self, id: NodeId) -> StackAncestorIdsIter<'_, 'a> {
        StackAncestorIdsIter { stack: self.stack.as_slice(), index: self.position(id) }
    }
}

/// Iterator over the ids of a node's ancestors in [`AncestorStack`].
///
/// Yields the parent first and the root ([`Program`]) last, matching the order
/// used by full [`AstNodes`] storage.
///
/// [`Program`]: oxc_ast::ast::Program
#[derive(Clone)]
pub struct StackAncestorIdsIter<'n, 'a> {
    stack: &'n [StackEntry<'a>],
    /// Position of the node whose ancestors are being yielded. Walks downward.
    index: usize,
}

impl Iterator for StackAncestorIdsIter<'_, '_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == 0 {
            // Root has no parent.
            return None;
        }
        self.index -= 1;
        Some(self.stack[self.index].id)
    }
}

/// Dynamic AST node storage. See the [module docs](self).
pub enum NodeStorage<'a> {
    /// Full random-access storage, retained after the build.
    Full(AstNodes<'a>),
    /// Only the live ancestor chain is retained during the build.
    Ancestors(AncestorStack<'a>),
}

impl Default for NodeStorage<'_> {
    fn default() -> Self {
        NodeStorage::Full(AstNodes::default())
    }
}

impl<'a> NodeStorage<'a> {
    /// Create full random-access storage.
    pub fn full() -> Self {
        NodeStorage::Full(AstNodes::default())
    }

    /// Create lightweight parent-pointer (ancestor stack) storage.
    pub fn ancestor_stack() -> Self {
        NodeStorage::Ancestors(AncestorStack::default())
    }

    /// `true` if this is [`NodeStorage::Full`] (random-access node storage).
    #[inline]
    pub fn is_full(&self) -> bool {
        matches!(self, NodeStorage::Full(_))
    }

    /// Consume the storage, returning the recorded [`AstNodes`].
    ///
    /// In ancestor-stack mode the chain is empty by the end of traversal, so
    /// this returns an empty [`AstNodes`].
    pub fn into_ast_nodes(self) -> AstNodes<'a> {
        match self {
            NodeStorage::Full(nodes) => nodes,
            NodeStorage::Ancestors(_) => AstNodes::default(),
        }
    }

    pub fn add_node(
        &mut self,
        kind: AstKind<'a>,
        scope_id: ScopeId,
        parent_node_id: NodeId,
        #[cfg(feature = "cfg")] cfg_id: BlockNodeId,
        flags: NodeFlags,
    ) -> NodeId {
        match self {
            NodeStorage::Full(nodes) => nodes.add_node(
                kind,
                scope_id,
                parent_node_id,
                #[cfg(feature = "cfg")]
                cfg_id,
                flags,
            ),
            NodeStorage::Ancestors(stack) => stack.add_node(kind, scope_id, flags),
        }
    }

    pub fn add_program_node(
        &mut self,
        kind: AstKind<'a>,
        scope_id: ScopeId,
        #[cfg(feature = "cfg")] cfg_id: BlockNodeId,
        flags: NodeFlags,
    ) -> NodeId {
        match self {
            NodeStorage::Full(nodes) => nodes.add_program_node(
                kind,
                scope_id,
                #[cfg(feature = "cfg")]
                cfg_id,
                flags,
            ),
            NodeStorage::Ancestors(stack) => stack.add_program_node(kind, scope_id, flags),
        }
    }

    /// Pop the current node and return its parent's id.
    #[inline]
    pub fn pop_node(&mut self, current_node_id: NodeId) -> NodeId {
        match self {
            NodeStorage::Full(nodes) => nodes.parent_id(current_node_id),
            NodeStorage::Ancestors(stack) => stack.pop_node(),
        }
    }

    #[inline]
    pub fn get_node(&self, id: NodeId) -> &AstNode<'a> {
        match self {
            NodeStorage::Full(nodes) => nodes.get_node(id),
            NodeStorage::Ancestors(stack) => stack.get_node(id),
        }
    }

    #[inline]
    pub fn kind(&self, id: NodeId) -> AstKind<'a> {
        self.get_node(id).kind()
    }

    #[inline]
    pub fn parent_id(&self, id: NodeId) -> NodeId {
        match self {
            NodeStorage::Full(nodes) => nodes.parent_id(id),
            NodeStorage::Ancestors(stack) => stack.parent_id(id),
        }
    }

    #[inline]
    pub fn parent_kind(&self, id: NodeId) -> AstKind<'a> {
        self.kind(self.parent_id(id))
    }

    #[inline]
    pub fn parent_node(&self, id: NodeId) -> &AstNode<'a> {
        self.get_node(self.parent_id(id))
    }

    #[inline]
    pub fn flags_mut(&mut self, id: NodeId) -> &mut NodeFlags {
        match self {
            NodeStorage::Full(nodes) => nodes.flags_mut(id),
            NodeStorage::Ancestors(stack) => stack.flags_mut(id),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            NodeStorage::Full(nodes) => nodes.len(),
            NodeStorage::Ancestors(stack) => stack.len as usize,
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        // Only full storage grows with the node count; the ancestor stack only
        // ever holds the current nesting depth.
        if let NodeStorage::Full(nodes) = self {
            nodes.reserve(additional);
        }
    }

    /// Walk up the AST, iterating over each parent [`NodeId`].
    ///
    /// The first id produced is the parent of `id`; the last is always the
    /// root [`Program`].
    ///
    /// [`Program`]: oxc_ast::ast::Program
    #[inline]
    pub fn ancestor_ids(&self, id: NodeId) -> impl Iterator<Item = NodeId> + Clone + '_ {
        match self {
            NodeStorage::Full(nodes) => Either::Left(nodes.ancestor_ids(id)),
            NodeStorage::Ancestors(stack) => Either::Right(stack.ancestor_ids(id)),
        }
    }

    /// Walk up the AST, iterating over each parent [`AstKind`].
    #[inline]
    pub fn ancestor_kinds(&self, id: NodeId) -> impl Iterator<Item = AstKind<'a>> + Clone + '_ {
        self.ancestor_ids(id).map(move |id| self.kind(id))
    }

    /// Walk up the AST, iterating over each parent [`AstNode`].
    #[inline]
    pub fn ancestors(&self, id: NodeId) -> impl Iterator<Item = &AstNode<'a>> + Clone + '_ {
        self.ancestor_ids(id).map(move |id| self.get_node(id))
    }

    /// Walk up the AST, iterating over each parent [`NodeId`] and [`AstNode`].
    #[inline]
    pub fn ancestors_enumerated(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = (NodeId, &AstNode<'a>)> + Clone + '_ {
        self.ancestor_ids(id).map(move |id| (id, self.get_node(id)))
    }
}
