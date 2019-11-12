{-# LANGUAGE AllowAmbiguousTypes       #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE OverloadedStrings         #-}
{-# LANGUAGE RankNTypes                #-}
{-# LANGUAGE UnicodeSyntax             #-}
{-# LANGUAGE QuasiQuotes #-}

import           Control.Applicative
import           Control.Monad                  (when)
import           Control.Monad.State
import           Data.Char                      (toLower)
import           Data.List
import qualified Data.Map.Strict                as M
import           Data.Maybe
import           Data.String
import           Data.Typeable
import           Debug.Trace
import qualified Penrose.Element                as D
import           Penrose.Env                    hiding (typeName)
import           Penrose.Pretty
import           Penrose.Substance
import           Penrose.Util                   (pickOne)
import           System.Environment
import           System.Directory (createDirectoryIfMissing)
import           System.Exit
import           System.IO
import           System.IO.Unsafe
import           System.Random
import           Text.PrettyPrint.HughesPJClass
import System.Console.Docopt
import           System.Console.Pretty (Color (..), Style (..), bgColor, color,
                                        style, supportsPretty)


-----------------------------  Randomness Definitions --------------------------
type Interval = (Int, Int)

{-# NOINLINE seedRnd #-}
seedRnd :: Int
seedRnd = 7

rndNum :: Interval -> Synthesize Int
rndNum interval = do
  (n, g') <- gets (randomR interval . gen)
  modify $ \c -> c {gen = g'}
  return n

choice :: [a] -> Synthesize a
choice lst = do
  (i, g') <- gets (randomR (0, length lst - 1) . gen)
  modify $ \c -> c {gen = g'}
  return $ lst !! i

choiceSafe :: [a] -> Synthesize (Maybe a)
choiceSafe [] = return Nothing
choiceSafe lst = do
  (i, g') <- gets (randomR (0, length lst - 1) . gen)
  modify $ \c -> c {gen = g'}
  return $ Just (lst !! i)

--------------------------------------------------------------------------------
-- Synthesizer context
type Name = String

type Names = M.Map String Int

type Synthesize a = State Context a

data GenericOption = Concrete | General VarEnv

data Context = Context
  { names           :: Names
  , declaredTypes   :: M.Map String [Name] -- | Map from type name to a list of names with the type
  , prog            :: SubProg -- | AST of the generated program
  , gen             :: StdGen -- | A random generator which will be updated regulary
  , seed            :: Int -- | random seed
  -- , nestingLevel    :: Int -- | the nesting level of the system in a current position
  } deriving (Show)


initContext :: VarEnv -> Context
initContext domainEnv =
  Context
  { declaredTypes = M.empty
  , names = M.empty
  , gen = mkStdGen seedRnd
  , seed = seedRnd
  , prog = []
  -- , nestingLevel = 0
  }

-- | Add statement to the AST
appendStmt :: SubStmt -> Synthesize ()
appendStmt stmt = modify $ \cxt -> cxt {prog = prog cxt ++ [stmt]}

--------------------------------------------------------------------------------
-- CLI

argPatterns :: Docopt
argPatterns = [docoptFile|penrose-synthesizer/USAGE.txt|]

getArgOrExit = getArgOrExitWith argPatterns

--------------------------------------------------------------------------------
-- The Substance synthesizer 

-- | The main function of the genrator.
main :: IO ()
main = do
  -- Read command line options
  args <- parseArgsOrExit argPatterns =<< getArgs
  domainFile <- args `getArgOrExit` argument "domain"
  -- substance <- args `getArgOrExit` longOption "substance"
  -- style <- args `getArgOrExit` longOption "style"
  path <- args `getArgOrExit` longOption "path"
  numProgs <- args `getArgOrExit` longOption "num-programs" 
  let n = read numProgs :: Int -- convert to int
  createDirectoryIfMissing True path -- create output dir if missing

  domainIn <- readFile domainFile
  let domainEnv = D.parseElement domainFile domainIn
  case domainEnv of
    Left err -> error $ show err
    Right domainEnv -> do
      let files = map (\i -> path ++ "/prog-" ++ show i ++ ".sub") [1 .. n]
      foldM_ (go domainEnv) (initContext domainEnv) files
  where 
    go env cxt file = do
      putStrLn (bgColor Red $ "Generated new program (" ++ file ++ "): ")
      let cxt' = initContext env 
      let (prog, cxt'') = runState (generateProgram env) (cxt' {gen = gen cxt})
      putStrLn prog
      writeFile file prog
      return cxt''

-- | The top level function for automatic generation of substance programs,
--   calls other functions to generate specific statements
generateProgram :: VarEnv -> Synthesize String
generateProgram domainEnv = do
  i <- rndNum (1, 2) -- FIXME: get rid of the hard-coded numbers
  j <- rndNum (1, 4) -- FIXME: get rid of the hard-coded numbers
  generateTypes domainEnv i
  generateStatements domainEnv j
  prog <- gets prog
  return $ render $ prettySubstance prog

-- | Generate random Substance statements
generateStatements :: VarEnv -> Int -> Synthesize ()
generateStatements domainEnv n = replicateM_ n (generateStatement domainEnv)

-- | Generate single random Substance statement
generateStatement :: VarEnv -> Synthesize ()
generateStatement domainEnv = do
  stmtF <- choice stmts
  stmt <- stmtF domainEnv
  appendStmt stmt
  where
    stmts =
      [ generatePredicate 
        -- generateBinding
      -- , generateValueBinding domainEnv context1
      ]

-- zeroNestingLevel( generatePredicateEquality domainEnv context1)
-- | Generate object declarations
generateTypes :: VarEnv -> Int -> Synthesize [SubStmt]
generateTypes domainEnv n = replicateM n (generateType domainEnv)

-- | Generate a single object declaration
generateType :: VarEnv -> Synthesize SubStmt
generateType domainEnv = do
  let types = M.toList (typeConstructors domainEnv)
  (typ, _) <- choice types -- COMBAK: check for empty list
  generateType' typ Concrete

generateType' :: String -> GenericOption -> Synthesize SubStmt
generateType' typ Concrete = do
  name <- freshName typ
  let stmt = Decl (TConstr $ TypeCtorApp {nameCons = typ, argCons = []}) (VarConst name)
  appendStmt stmt -- TODO: find a less redundant way to synthesize the AST
  return stmt
generateType' typ (General env) = do
  name <- freshName typ
  let types = possibleTypes env typ
  typ' <- choice types 
  generateType' typ' Concrete

-- | Generate a single predicate
-- FIXME: currently not handling nesting
generatePredicate :: VarEnv -> Synthesize SubStmt
generatePredicate domainEnv = do
  let preds = M.toList (predicates domainEnv)
  (_, p) <- choice preds
  gen p
  where
    gen (Pred1 p1) = generatePredicate1 domainEnv
    gen (Pred2 p2) = generatePredicate2 domainEnv

generatePredicate1, generatePredicate2 :: VarEnv -> Synthesize SubStmt
generatePredicate1 domainEnv = do
  pred <- choice (pred1s domainEnv)
  args <- generatePredArgs domainEnv (map typeName $ tlspred1 pred)
  let stmt = ApplyP $ Predicate { predicateName = PredicateConst $ namepred1 pred, predicateArgs = args }
  return stmt
-- TODO: make sure pred2 is higher-level predicates?
generatePredicate2 domainEnv = do
  pred <- choice (pred2s domainEnv)
  let args = []
  return $ ApplyP $ Predicate { predicateName = PredicateConst $ namepred2 pred, predicateArgs = args }

-- | Get a list of all the types needed for a specific statement and
--   add type declrations for them in case needed
generatePredArgs :: VarEnv -> [String] -> Synthesize [PredArg]
generatePredArgs domainEnv = mapM gen 
  where 
    gen t = do
      existingTypes <- gets declaredTypes
      case M.lookup t existingTypes of
        Nothing -> do 
          generateType' t Concrete -- FIXME: concrete types for now
          gen t -- NOTE: slightly costly to do the extra loopup. Optimize when needed
        Just lst -> do
          n <- choice lst -- pick one existing id
          return $ PE $ VarE $ VarConst n

-- generatePredicate2 :: VarEnv -> Context -> Context
-- generatePredicate2 domainEnv context
--             --allPreds = M.toAscList (predicates domainEnv)
--  =
--   let g = mkStdGen 9
--       l = length (pred2Lst context) - 1
--       p = fst (randomR (0, l) g)
--       predicate = pred2Lst context !! p
--       context1 = context {prog = prog context ++ namepred2 predicate ++ "("}
--       n = length (plspred2 predicate)
--       context2 = generatePred2Args domainEnv context1 n
--   in context2
-- | Generate single random binding statement
-- FIXME: this function seems to generate types retroactively... We shouldn't do that in general??
-- generateBinding :: VarEnv -> Context -> Context
-- generateBinding domainEnv context = do
--   let operations = M.toAscList (operators domainEnv)
--   operation <- choice operations
--   allTypes = top operation : tlsop operation
--   generatePreDeclarations domainEnv (getTypeNames allTypes)
--              context3 = generateArgument context2 (convert (top operation))
--              context4 =
--                context3 {prog = prog context3 ++ " := " ++ nameop operation}
--          in generateArguments
--               domainEnv
--               (getTypeNames (tlsop operation))
--               context4
--     else context
-- -- | Generate single random binding statement
-- generateValueBinding :: VarEnv -> Context -> Context
-- generateValueBinding domainEnv context =
--   if not (null (valConstructors domainEnv))
--     then let valConsts = M.toAscList (valConstructors domainEnv)
--              (v, context1) = rndNum context (0, length valConsts - 1)
--              valConstructor = snd (valConsts !! v)
--              allTypes = tvc valConstructor : tlsvc valConstructor
--              context2 =
--                generatePreDeclarations
--                  domainEnv
--                  context1
--                  (getTypeNames allTypes)
--              context3 = generateArgument context2 (convert (tvc valConstructor))
--              context4 =
--                context3
--                {prog = prog context3 ++ " := " ++ namevc valConstructor}
--          in generateArguments
--               domainEnv
--               (getTypeNames (tlsvc valConstructor))
--               context4
--     else context
-- -- | Generate single random predicate equality
-- -- generatePredicateEquality :: VarEnv -> Context -> Context
-- -- generatePredicateEquality domainEnv context =
-- --   let context1 = generatePredicate domainEnv context False
-- --       context2 = context1 {prog = prog context1 ++ " <-> "}
-- --   in generatePredicate domainEnv context2 True
-- generatePred2Args :: VarEnv -> Context -> Int -> Context
-- generatePred2Args domainEnv context n =
--   let context1 = generatePred2Arg domainEnv context
--   in if n /= 1
--        then let context2 = context1 {prog = prog context1 ++ ","}
--             in generatePred2Args domainEnv context2 (n - 1)
--        else context1 {prog = prog context1 ++ ")"}
-- generatePred2Arg :: VarEnv -> Context -> Context
-- generatePred2Arg domainEnv context = generatePredicate domainEnv context True
--------------------------------------------------------------------------------
-- Helpers
pred1s :: VarEnv -> [Predicate1]
pred1s domainEnv = map (\(Pred1 p) -> p) $ M.elems $ predicates domainEnv

pred2s :: VarEnv -> [Predicate2]
pred2s domainEnv = map (\(Pred2 p) -> p) $ M.elems $ predicates domainEnv

possibleTypes :: VarEnv -> String -> [String]
possibleTypes domainEnv t =
  let subt = subTypes domainEnv
      allTypes = [typeName t1 | (t1, t2) <- subt, typeName t2 == t]
  in (t : allTypes)

typeName :: T -> String
typeName (TTypeVar t) = typeVarName t
typeName (TConstr t)  = nameCons t


-- -- | Get an argument for operations / predicates / val constructors
-- generateArgument :: Context -> String -> Context
-- generateArgument context t =
--   case M.lookup t (declaredTypes context) of
--     Nothing -> error "Generation Error!"
--     Just lst ->
--       let (a, context1) = rndNum context (0, length lst - 1)
--       in context1 {prog = prog context1 ++ lst !! a}
-- -- | Get an argument for operations / predicates / val constructors
-- generateFullArgument :: VarEnv -> Context -> String -> Context
-- generateFullArgument domainEnv context t =
--   let allPossibleArguments = getAllPossibleArguments domainEnv context t
--       (a, context1) = rndNum context (0, length allPossibleArguments - 1)
--   in context1 {prog = prog context1 ++ allPossibleArguments !! a ++ ","}
-- getAllPossibleArguments :: VarEnv -> Context -> String -> [String]
-- getAllPossibleArguments domainEnv context t =
--   concatMap (getIdentifiers context) (getAllPossibleTypes domainEnv context t)

-- getIdentifiers :: Context -> String -> [String]
-- getIdentifiers context t = fromMaybe [] (M.lookup t (declaredTypes context))
--------------------------------------------------------------------------------
-- Names generation
-- | Generate a new name given a type
freshName :: String -> Synthesize String
freshName typ = do
  cxt <- get
  let (n, names') = uniqueName (prefixOf typ) $ names cxt
  modify $ \cxt ->
    cxt
    { declaredTypes = M.insertWith (++) typ [n] (declaredTypes cxt)
    , names = names'
    }
  return n

prefixOf :: String -> String
prefixOf = map toLower . take 1

uniqueName :: String -> Names -> (String, Names)
uniqueName nm ns =
  case M.lookup nm ns of
    Nothing -> (nm, M.insert nm 1 ns)
    Just ix -> (nm ++ show ix, M.insert nm (ix + 1) ns)